#!/usr/bin/env python3
"""
일간 RS 모니터링 텔레그램봇
- 20종목 워치리스트 + 4종목 벤치마크 추적
- IBD RS, 2/28 대비 누적 등락률, 거래량 품질, 신고가 여부
- WTI 트리거 모니터링
"""
import os, time
import numpy as np
import pandas as pd
import datetime as dt
import requests
import warnings
warnings.filterwarnings("ignore")

try:
    import FinanceDataReader as fdr
    from pykrx import stock as krx
except ImportError:
    raise

# ── 핵심 5종목 (Tier 1: 반도체 소부장 + Tier 2: 통신/RF) ──
# ── 비교군 5종목 (같은 섹터 차선) ──
# ── 보유 3종목 ──
# ── 보류 3종목 (Tier 3) ──
WATCHLIST = {
    "036930": "주성엔지니어링",
    "058470": "리노공업",
    "031980": "피에스케이홀딩스",
    "218410": "RFHIC",
    "138080": "오이솔루션",
    "095340": "ISC",
    "039030": "이오테크닉스",
    "357780": "솔브레인",
    "327260": "RF머트리얼즈",
    "010170": "대한광통신",
    "005930": "삼성전자",
    "069500": "KODEX 200",
    "229200": "KODEX 코스닥150",
    "272210": "한화시스템",
    "131290": "티에스이",
    "079550": "LIG넥스원",
}

BENCHMARKS = {
    "KS11":   "코스피지수",
    "KQ11":   "코스닥지수",
    "475310": "SOL 반도체후공정",
    "261220": "KODEX WTI원유선물(H)",
}

# 종목 그룹 분류 (리포트 출력용)
GROUP = {}
for c in ["036930","058470","031980","218410","138080"]:
    GROUP[c] = "핵심"
for c in ["095340","039030","357780","327260","010170"]:
    GROUP[c] = "비교"
for c in ["005930","069500","229200"]:
    GROUP[c] = "보유"
for c in ["272210","131290","079550"]:
    GROUP[c] = "보류"

# 2/28 기준일 (RS 추적 기준)
BASELINE_DATE = "20260228"

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
QUARTER = 63


# ──────────────────────────────────────────────
# 네이버 금융 API로 전종목 리스트 수집 (KRX API 장애 대응)
# ──────────────────────────────────────────────
def get_all_tickers():
    """코스피+코스닥 전종목 가져오기 (네이버 금융 모바일 API)"""
    headers = {"User-Agent": "Mozilla/5.0"}
    all_rows = []
    for market in ["KOSPI", "KOSDAQ"]:
        page = 1
        page_size = 100
        count = 0
        while True:
            url = f"https://m.stock.naver.com/api/stocks/marketValue/{market}?page={page}&pageSize={page_size}"
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()
            stocks = data.get("stocks", [])
            if not stocks:
                break
            for s in stocks:
                code = s.get("itemCode", "")
                name = s.get("stockName", "")
                if len(code) == 6:
                    all_rows.append({"Code": code, "Name": name})
            count += len(stocks)
            if count >= data.get("totalCount", 0):
                break
            page += 1
    return pd.DataFrame(all_rows).dropna(subset=["Code", "Name"])


def get_tickers_safe():
    """fdr.StockListing 시도 -> 실패 시 네이버 API fallback"""
    try:
        kospi  = fdr.StockListing("KOSPI")[["Code","Name"]]
        kosdaq = fdr.StockListing("KOSDAQ")[["Code","Name"]]
        tickers = pd.concat([kospi, kosdaq], ignore_index=True)
        if len(tickers) < 100:
            raise ValueError("KRX API returned too few tickers")
        print(f"[종목리스트] fdr.StockListing: {len(tickers)}개")
        return tickers
    except Exception as e:
        print(f"[종목리스트] fdr.StockListing 실패 ({e}), 네이버 API 사용")
        tickers = get_all_tickers()
        print(f"[종목리스트] 네이버 API: {len(tickers)}개")
        return tickers


# ──────────────────────────────────────────────
# pykrx 비거래일 대응: 최근 거래일 자동 탐색
# ──────────────────────────────────────────────
def find_last_trading_date(ref_date=None):
    """ref_date(YYYYMMDD)로부터 역순 탐색하여 최근 거래일 반환"""
    if ref_date is None:
        ref_date = dt.date.today()
    elif isinstance(ref_date, str):
        ref_date = dt.datetime.strptime(ref_date, "%Y%m%d").date()

    for delta in range(7):
        d = ref_date - dt.timedelta(days=delta)
        if d.weekday() >= 5:  # 주말 스킵
            continue
        ds = d.strftime("%Y%m%d")
        try:
            df = krx.get_market_ohlcv_by_date(ds, ds, "005930")
            if df is not None and not df.empty:
                return ds
        except Exception:
            pass
    # 못 찾으면 원래 날짜 반환
    return ref_date.strftime("%Y%m%d") if isinstance(ref_date, dt.date) else ref_date


# ──────────────────────────────────────────────
# 지수 데이터 (KRX API 장애 대응: 네이버 차트 API fallback)
# ──────────────────────────────────────────────
INDEX_NAVER_MAP = {
    "KS11": "KOSPI",
    "KQ11": "KOSDAQ",
}

def get_index_prices_naver(symbol, start, end):
    """네이버 차트 API로 지수 가격 히스토리 수집"""
    naver_code = INDEX_NAVER_MAP.get(symbol)
    if not naver_code:
        return pd.Series(dtype=float)
    url = (f"https://api.stock.naver.com/chart/domestic/index/{naver_code}"
           f"?periodType=dayCandle&timeframe=day"
           f"&startDateTime={start}&endDateTime={end}")
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    data = r.json()
    infos = data.get("priceInfos", [])
    if not infos:
        return pd.Series(dtype=float)
    rows = [{"Date": pd.Timestamp(p["localDate"]), "Close": p["closePrice"]} for p in infos]
    df = pd.DataFrame(rows).set_index("Date").sort_index()
    return df["Close"]


def get_prices_safe(code, start, end):
    """fdr.DataReader 시도 -> 지수의 경우 실패 시 네이버 차트 API fallback"""
    # 지수가 아닌 일반 종목/ETF는 fdr.DataReader 사용
    if code not in INDEX_NAVER_MAP:
        return fdr.DataReader(code, start, end)

    # 지수: fdr 시도 -> 실패 시 네이버 fallback
    try:
        df = fdr.DataReader(code, start, end)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    print(f"[지수 fallback] {code} -> 네이버 차트 API 사용")
    prices = get_index_prices_naver(code, start, end)
    if prices.empty:
        return pd.DataFrame()
    # fdr.DataReader와 유사한 형태로 반환
    return pd.DataFrame({"Close": prices})


# ──────────────────────────────────────────────
# RS 계산
# ──────────────────────────────────────────────
def calc_rs_score(prices):
    if len(prices) < QUARTER * 4:
        return np.nan
    def qr(s, e):
        return prices.iloc[-e] / prices.iloc[-s] - 1
    try:
        return qr(QUARTER,1)*2 + qr(QUARTER*2,QUARTER) + qr(QUARTER*3,QUARTER*2) + qr(QUARTER*4,QUARTER*3)
    except Exception:
        return np.nan


def build_universe_rs(start, end):
    cache_file = f"rs_cache_{end}.csv"
    if os.path.exists(cache_file):
        print(f"[캐시 로드] {cache_file}")
        return pd.read_csv(cache_file, dtype={"Code": str})

    tickers = get_tickers_safe()
    records = []
    total = len(tickers)
    for idx, row in tickers.iterrows():
        try:
            df = fdr.DataReader(row["Code"], start, end)
            if df.empty or len(df) < QUARTER*4:
                continue
            score = calc_rs_score(df["Close"])
            if not np.isnan(score):
                records.append({"Code": row["Code"], "Name": row["Name"], "Score": score})
            time.sleep(0.05)
        except Exception:
            pass
        if (idx + 1) % 500 == 0:
            print(f"  진행: {idx+1}/{total}")

    u = pd.DataFrame(records)
    if u.empty:
        print("[경고] RS 계산 결과 0건")
        return u
    u["Rank"] = u["Score"].rank(ascending=True)
    u["RS"]   = ((u["Rank"]-1)/(len(u)-1)*98+1).astype(int)
    u.to_csv(cache_file, index=False)
    print(f"[완료] {len(u)}개 종목 -> {cache_file}")
    return u


def rs_line_slope(prices, kospi, window=20):
    a, b = prices.align(kospi, join="inner")
    rs = a / b
    if len(rs) < window+1:
        return np.nan
    return round((rs.iloc[-1]/rs.iloc[-window]-1)*100, 2)


def minervini_check(prices):
    if len(prices) < 260:
        return False
    c = prices.iloc[-1]
    ma50  = prices.tail(50).mean()
    ma150 = prices.tail(150).mean()
    ma200 = prices.tail(200).mean()
    prev200 = prices.tail(220).head(200).mean()
    hi52 = prices.tail(260).max()
    lo52 = prices.tail(260).min()
    return (c > ma50 > ma150 > ma200 > prev200
            and c > lo52 * 1.30
            and c >= hi52 * 0.75)


# ──────────────────────────────────────────────
# 새 지표: 2/28 대비 누적 등락률, 거래량 품질, 신고가
# ──────────────────────────────────────────────
def calc_change_from_baseline(prices, baseline_date=BASELINE_DATE):
    """2/28 종가 대비 현재 등락률(%)"""
    try:
        bd = pd.Timestamp(baseline_date)
        # baseline_date 이전 가장 가까운 거래일 찾기
        mask = prices.index <= bd
        if mask.any():
            base_price = prices[mask].iloc[-1]
            return round((prices.iloc[-1] / base_price - 1) * 100, 2)
    except Exception:
        pass
    return np.nan


def calc_volume_quality(df, window=20):
    """거래량 품질: 상승일 평균거래량 / 하락일 평균거래량 (>1 = 건강)"""
    if len(df) < window:
        return np.nan
    recent = df.tail(window).copy()
    recent["Change"] = recent["Close"].pct_change()
    up_vol   = recent.loc[recent["Change"] > 0, "Volume"].mean()
    down_vol = recent.loc[recent["Change"] < 0, "Volume"].mean()
    if pd.isna(down_vol) or down_vol == 0:
        return np.nan
    return round(up_vol / down_vol, 2)


def check_new_high(prices, lookback=260):
    """52주 신고가 여부"""
    if len(prices) < 2:
        return False
    hi = prices.tail(lookback).max()
    return prices.iloc[-1] >= hi


# ──────────────────────────────────────────────
# 기관 수급
# ──────────────────────────────────────────────
def get_inst_net_buy(date_str):
    """기관 순매수 조회. KRX API 장애 시 None 반환."""
    try:
        ds = find_last_trading_date(date_str)
        df = krx.get_market_trading_value_by_date(ds, ds, "KOSPI")
        if df is not None and not df.empty:
            return df["기관합계"].iloc[-1] / 1e8
    except Exception:
        pass
    return None


def get_inst_trend(days=5):
    """최근 N거래일 기관 순매수 추이. KRX 장애 시 빈 리스트."""
    results, today, checked, delta = [], dt.date.today(), 0, 0
    while checked < days and delta < 30:
        d = today - dt.timedelta(days=delta)
        delta += 1
        if d.weekday() >= 5:
            continue
        try:
            ds = d.strftime("%Y%m%d")
            df = krx.get_market_trading_value_by_date(ds, ds, "KOSPI")
            if df is not None and not df.empty:
                results.append((d.strftime("%m/%d"), df["기관합계"].iloc[-1] / 1e8))
                checked += 1
        except Exception:
            pass
    return list(reversed(results))


# ──────────────────────────────────────────────
# WTI 유가 확인
# ──────────────────────────────────────────────
def get_wti_price():
    """WTI 최근 종가 조회 (fdr 사용)"""
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=10)
        df = fdr.DataReader("CL=F", start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
        if df is not None and not df.empty:
            return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return None


def wti_trigger_text(wti):
    """WTI 트리거 판정 텍스트"""
    if wti is None:
        return "[!] WTI 조회 실패"
    if wti >= 85:
        return f"[!!] WTI ${wti} >= $85 -> 매수 전면 중단"
    elif wti >= 80:
        return f"[!] WTI ${wti} >= $80 -> 매수 감속 주의"
    else:
        return f"[OK] WTI ${wti} < $80 -> 매수 가능"


# ──────────────────────────────────────────────
# 텔레그램 전송
# ──────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(msg)
        return
    # 텔레그램 메시지 길이 제한 (4096자)
    for i in range(0, len(msg), 4000):
        chunk = msg[i:i+4000]
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"},
                timeout=10,
            )
        except Exception as e:
            print(f"[텔레그램 전송 실패] {e}")


# ──────────────────────────────────────────────
# 메인 모드: full (장 마감 후)
# ──────────────────────────────────────────────
def run_full():
    today = dt.date.today()
    end_str   = today.strftime("%Y%m%d")
    start_str = (today - dt.timedelta(days=420)).strftime("%Y%m%d")

    # 벤치마크 가격
    kospi_prices = None
    bench_results = []
    for code, name in BENCHMARKS.items():
        try:
            df = get_prices_safe(code, start_str, end_str)
            if df is not None and not df.empty:
                prices = df["Close"]
                if code == "KS11":
                    kospi_prices = prices
                chg_base = calc_change_from_baseline(prices)
                bench_results.append({"종목명": name, "코드": code,
                                      "종가": prices.iloc[-1],
                                      "2/28대비(%)": chg_base})
        except Exception as e:
            print(f"[벤치마크 {name}] 오류: {e}")

    if kospi_prices is None:
        print("[경고] 코스피 지수 로드 실패, RS라인 계산 불가")
        kospi_prices = pd.Series(dtype=float)

    # WTI 확인
    wti = get_wti_price()

    # 기관 수급 (당일 + 5일 추이)
    trading_date = find_last_trading_date(end_str)
    inst_net = get_inst_net_buy(trading_date)
    inst_trend = get_inst_trend(days=5)

    # 전종목 RS (캐시)
    universe = build_universe_rs(start_str, end_str)
    rs_lookup = dict(zip(universe["Code"], universe["RS"])) if not universe.empty else {}

    # 워치리스트 분석
    results = []
    for code, name in WATCHLIST.items():
        try:
            df = fdr.DataReader(code, start_str, end_str)
            if df is None or df.empty:
                continue
            prices = df["Close"]
            vol_avg = df["Volume"].tail(20).mean()
            last_vol = df["Volume"].iloc[-1]
            vol_ratio = round(last_vol / vol_avg, 2) if vol_avg > 0 else np.nan

            # 등락률 (전일 대비)
            daily_chg = round((prices.iloc[-1] / prices.iloc[-2] - 1) * 100, 2) if len(prices) >= 2 else np.nan

            results.append({
                "그룹": GROUP.get(code, ""),
                "종목명": name,
                "코드": code,
                "종가": prices.iloc[-1],
                "전일비(%)": daily_chg,
                "2/28대비(%)": calc_change_from_baseline(prices),
                "RS": rs_lookup.get(code, None),
                "RS라인(20일%)": rs_line_slope(prices, kospi_prices),
                "미너비니": "V" if minervini_check(prices) else "-",
                "거래량비율": vol_ratio,
                "거래량품질": calc_volume_quality(df),
                "신고가": "H" if check_new_high(prices) else "-",
                "52주고점비(%)": round((prices.iloc[-1] / prices.tail(260).max() - 1) * 100, 1),
            })
        except Exception as e:
            print(f"[{name}] 오류: {e}")

    df_r = pd.DataFrame(results)
    if df_r.empty:
        send_telegram("[!] 워치리스트 데이터 수집 실패")
        return

    # 그룹별 정렬
    group_order = {"핵심": 0, "비교": 1, "보유": 2, "보류": 3}
    df_r["_sort"] = df_r["그룹"].map(group_order).fillna(9)
    df_r = df_r.sort_values(["_sort", "RS"], ascending=[True, False]).drop(columns=["_sort"])

    # CSV 저장
    csv_name = f"rs_report_{end_str}.csv"
    df_r.to_csv(csv_name, index=False, encoding="utf-8-sig")
    print(f"[저장] {csv_name}")

    # 텔레그램 메시지 구성
    if inst_net is not None:
        inst_icon = "+" if inst_net > 0 else "-"
        inst_str = f"기관 순매수(당일): {inst_icon}{abs(inst_net):.0f}억"
    else:
        inst_str = "기관 순매수(당일): 조회불가 (KRX API 장애)"
    lines = [
        f"*RS 마감 리포트 [{today.strftime('%m/%d')}]*",
        inst_str,
        wti_trigger_text(wti),
        "",
        "*최근 5일 기관 순매수 추이*",
    ]
    if inst_trend:
        for d, v in inst_trend:
            icon = "+" if v > 0 else "-"
            lines.append(f"  {d}: {icon}{abs(v):.0f}억")
    else:
        lines.append("  조회불가 (KRX API 장애)")
    lines.append("")

    # 벤치마크
    lines.append("*[벤치마크]*")
    for b in bench_results:
        chg = b["2/28대비(%)"]
        chg_str = f"{chg:+.1f}%" if not pd.isna(chg) else "N/A"
        lines.append(f"  {b['종목명']}: {b['종가']:,.0f}  (2/28비: {chg_str})")
    lines.append("")

    # 워치리스트
    current_group = None
    for _, r in df_r.iterrows():
        grp = r["그룹"]
        if grp != current_group:
            current_group = grp
            lines.append(f"*[{grp}]*")

        rs = r["RS"]
        if rs and rs >= 90:
            flag = "[**]"
        elif rs and rs >= 80:
            flag = "[*]"
        else:
            flag = "[ ]"

        vol_dir = ""
        vr = r["거래량비율"]
        dc = r["전일비(%)"]
        if not pd.isna(vr) and not pd.isna(dc):
            if dc > 0 and vr >= 1.5:
                vol_dir = " 상승+거래량+"
            elif dc < 0 and vr >= 1.5:
                vol_dir = " 하락+거래량+"
            elif dc < 0 and vr < 0.8:
                vol_dir = " 건강조정"

        nh = " NEW-HIGH" if r["신고가"] == "H" else ""
        chg_base = r["2/28대비(%)"]
        chg_base_str = f"{chg_base:+.1f}%" if not pd.isna(chg_base) else "N/A"

        lines.append(
            f"{flag} *{r['종목명']}* RS={rs} "
            f"전일{r['전일비(%)']:+.1f}% "
            f"2/28비:{chg_base_str} "
            f"VQ:{r['거래량품질']} "
            f"Vol:{vr}x "
            f"{r['미너비니']}{nh}{vol_dir}"
        )

    send_telegram("\n".join(lines))
    print(df_r.to_string(index=False))


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────
def main():
    print("[실행] RS 마감 리포트")
    run_full()


if __name__ == "__main__":
    main()
