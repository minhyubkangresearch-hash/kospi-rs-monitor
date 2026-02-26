#!/usr/bin/env python3
"""
KOSPI Daily RS Monitor v2
- 오전 8:50 KST: 전일 RS 리포트 재전송 + 기관 수급 방향
- 오후 4:30 KST: 당일 RS 신규 계산 (메인 실행)
- 장 중 13:30 KST: 기관 순매수 금액만 체크
"""

import os
import time
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
    print("pip install finance-datareader pykrx requests 를 먼저 실행하세요.")
    raise

WATCHLIST = {
    #코스피
 WATCHLIST = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "051910": "LG화학",
    "207940": "삼성바이오로직스",
    "005380": "현대차",
    "000270": "기아",
    "006800": "미래에셋증권",
    "066570": "LG전자",
    "069500": "KODEX200",
    "229200": "KODEX코스닥150",
    "328130": "루닛",
    "319400": "현대무벡스",
    "475830": "오름테라퓨틱",
}


TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
RUN_MODE         = os.environ.get("RUN_MODE", "full")

QUARTER = 63

def calc_rs_score(prices):
    n = len(prices)
    if n < QUARTER * 4:
        return np.nan
    def qr(s, e):
        return prices.iloc[-e] / prices.iloc[-s] - 1
    try:
        q1 = qr(QUARTER,         1)
        q2 = qr(QUARTER * 2,     QUARTER)
        q3 = qr(QUARTER * 3, QUARTER * 2)
        q4 = qr(QUARTER * 4, QUARTER * 3)
        return q1 * 2 + q2 + q3 + q4
    except (IndexError, ZeroDivisionError):
        return np.nan

def build_universe_rs(start, end, sleep=0.05):
    cache_file = f"rs_cache_{end}.csv"
    if os.path.exists(cache_file):
        print(f"[캐시 로드] {cache_file}")
        return pd.read_csv(cache_file, dtype={"Code": str})
    kospi   = fdr.StockListing("KOSPI")[["Code", "Name"]]
    kosdaq  = fdr.StockListing("KOSDAQ")[["Code", "Name"]]
    etf     = fdr.StockListing("ETF/KR")[["Code", "Name"]]
    tickers = pd.concat([kospi, kosdaq, etf], ignore_index=True)
    records = []
    for _, row in tickers.iterrows():
        code, name = row["Code"], row["Name"]
        try:
            df = fdr.DataReader(code, start, end)
            if df.empty or len(df) < QUARTER * 4:
                continue
            score = calc_rs_score(df["Close"])
            if not np.isnan(score):
                records.append({"Code": code, "Name": name, "Score": score})
            time.sleep(sleep)
        except Exception:
            pass
    universe = pd.DataFrame(records)
    universe["Rank"] = universe["Score"].rank(ascending=True)
    universe["RS"]   = ((universe["Rank"] - 1) / (len(universe) - 1) * 98 + 1).astype(int)
    universe.to_csv(cache_file, index=False)
    print(f"[완료] {len(universe)}개 종목 RS 계산 → {cache_file}")
    return universe

def rs_line_slope(prices, kospi, window=20):
    aligned = prices.align(kospi, join="inner")
    rs_line = aligned[0] / aligned[1]
    if len(rs_line) < window + 1:
        return np.nan
    return round((rs_line.iloc[-1] / rs_line.iloc[-window] - 1) * 100, 2)

def minervini_check(prices):
    if len(prices) < 260:
        return False
    c     = prices.iloc[-1]
    ma50  = prices.tail(50).mean()
    ma150 = prices.tail(150).mean()
    ma200 = prices.tail(200).mean()
    prev_ma200 = prices.tail(220).head(200).mean()
    hi52  = prices.tail(260).max()
    lo52  = prices.tail(260).min()
    return (
        c > ma50 > ma150 > ma200 > prev_ma200
        and c > lo52 * 1.30
        and c >= hi52 * 0.75
    )

def get_inst_net_buy(date_str):
    try:
        df = krx.get_market_trading_value_by_date(date_str, date_str, "KOSPI")
        if df.empty:
            return 0.0
        return df["기관합계"].iloc[-1] / 1e8
    except Exception:
        return 0.0

def get_inst_trend(days=5):
    results = []
    today = dt.date.today()
    checked = 0
    delta = 0
    while checked < days:
        d = today - dt.timedelta(days=delta)
        delta += 1
        if d.weekday() >= 5:
            continue
        date_str = d.strftime("%Y%m%d")
        try:
            df = krx.get_market_trading_value_by_date(date_str, date_str, "KOSPI")
            if not df.empty:
                val = df["기관합계"].iloc[-1] / 1e8
                results.append((d.strftime("%m/%d"), val))
                checked += 1
        except Exception:
            pass
    return list(reversed(results))

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram 미설정]\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    })
    if resp.status_code != 200:
        print(f"[Telegram 오류] {resp.text}")

def run_full():
    today     = dt.date.today()
    end_str   = today.strftime("%Y%m%d")
    start_str = (today - dt.timedelta(days=420)).strftime("%Y%m%d")
    kospi_df     = fdr.DataReader("KS11", start_str, end_str)
    kospi_prices = kospi_df["Close"]
    inst_net     = get_inst_net_buy(end_str)
    inst_icon    = "🟢" if inst_net > 0 else "🔴"
    print("RS 유니버스 계산 중...")
    universe  = build_universe_rs(start_str, end_str)
    rs_lookup = dict(zip(universe["Code"], universe["RS"]))
    results = []
    for code, name in WATCHLIST.items():
        try:
            df = fdr.DataReader(code, start_str, end_str)
            if df.empty:
                continue
            prices    = df["Close"]
            rs_rating = rs_lookup.get(code, None)
            slope     = rs_line_slope(prices, kospi_prices)
            trend_ok  = minervini_check(prices)
            hi52      = prices.tail(260).max()
            pct_hi    = round((prices.iloc[-1] / hi52 - 1) * 100, 1)
            vol       = df["Volume"].iloc[-1]
            vol_avg   = df["Volume"].tail(20).mean()
            vol_ratio = round(vol / vol_avg, 2) if vol_avg > 0 else np.nan
            results.append({
                "종목명": name, "RS": rs_rating,
                "RS라인(20일%)": slope, "미너비니": "✅" if trend_ok else "—",
                "52주고점대비(%)": pct_hi, "거래량비율": vol_ratio,
            })
        except Exception as e:
            print(f"  [{name}] 오류: {e}")
    df_result = pd.DataFrame(results).sort_values("RS", ascending=False)
    df_result.to_csv(f"rs_report_{end_str}.csv", index=False, encoding="utf-8-sig")
    lines = [f"📊 *KOSPI RS 마감 리포트 [{today.strftime('%m/%d')} 종가 기준]*",
             f"{inst_icon} 기관 순매수: {inst_net:+.0f}억", ""]
    for _, r in df_result.iterrows():
        rs   = r["RS"]
        flag = "🔥" if rs and rs >= 90 else ("✅" if rs and rs >= 80 else "⚪")
        lines.append(
            f"{flag} *{r['종목명']}*  RS={rs}  "
            f"RS라인: {r['RS라인(20일%)']:+.1f}%  "
            f"거래량: {r['거래량비율']}x  {r['미너비니']}"
        )
    send_telegram("\n".join(lines))
    print(df_result.to_string(index=False))

def run_morning():
    import glob
    today = dt.date.today()
    cache_files = sorted(glob.glob("rs_cache_*.csv"), reverse=True)
    if not cache_files:
        send_telegram("⚠️ 캐시 없음 — 오후 4:30 리포트를 먼저 실행하세요.")
        return
    cache_date = cache_files[0].replace("rs_cache_", "").replace(".csv", "")
    report_files = sorted(glob.glob("rs_report_*.csv"), reverse=True)
    if not report_files:
        send_telegram("⚠️ 리포트 없음 — 오후 4:30 리포트를 먼저 실행하세요.")
        return
    df_result = pd.read_csv(report_files[0])
    trend = get_inst_trend(days=5)
    trend_lines = []
    for date_label, val in trend:
        icon = "🟢" if val > 0 else "🔴"
        trend_lines.append(f"  {icon} {date_label}: {val:+.0f}억")
    lines = [
        f"🌅 *장 시작 브리핑 [{today.strftime('%m/%d')} 08:50]*",
        f"📋 기준: {cache_date[:4]}.{cache_date[4:6]}.{cache_date[6:]} 종가",
        "",
        "*최근 5일 기관 순매수 추이*",
    ] + trend_lines + ["", "*관심종목 RS 현황*", ""]
    for _, r in df_result.sort_values("RS", ascending=False).iterrows():
        rs   = r["RS"]
        flag = "🔥" if rs >= 90 else ("✅" if rs >= 80 else "⚪")
        lines.append(
            f"{flag} *{r['종목명']}*  RS={rs}  "
            f"RS라인: {r['RS라인(20일%)']:+.1f}%  {r['미너비니']}"
        )
    send_telegram("\n".join(lines))
    print("모닝 브리핑 전송 완료")

def run_intraday():
    today    = dt.date.today()
    end_str  = today.strftime("%Y%m%d")
    inst_net = get_inst_net_buy(end_str)
    inst_icon = "🟢" if inst_net > 0 else "🔴"
    if inst_net > 3000:
        signal = "💪 강한 유입 — 매매 적극 유지"
    elif inst_net > 0:
        signal = "👀 순유입 — 관망 유지"
    elif inst_net > -3000:
        signal = "⚠️ 소폭 이탈 — 주의"
    else:
        signal = "🚨 강한 이탈 — 포지션 점검 필요"
    now_kst = dt.datetime.utcnow() + dt.timedelta(hours=9)
    lines = [
        f"📡 *장 중 기관 수급 체크 [{now_kst.strftime('%m/%d %H:%M')} KST]*",
        f"{inst_icon} 기관 순매수(당일 누적): *{inst_net:+.0f}억*",
        f"→ {signal}",
        "",
        "_※ RS 수치는 종가 확정 후 오후 4:30 리포트를 참고하세요_"
    ]
    send_telegram("\n".join(lines))
    print("장 중 수급 체크 전송 완료")

def main():
    mode = RUN_MODE.strip().lower()
    print(f"[실행 모드] {mode}")
    if mode == "morning":
        run_morning()
    elif mode == "intraday":
        run_intraday()
    else:
        run_full()

if __name__ == "__main__":
    main()
