#!/usr/bin/env python3
"""
KOSPI Daily RS Monitor
- O'Neil 방식 RS Rating 계산 (전체 KOSPI 대비 백분위)
- 기관 순매수 현황 (pykrx)
- Telegram 알림
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

# ============================================================
# ① 설정: 관심종목 & Telegram 시크릿
# ============================================================
WATCHLIST = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "035420": "NAVER",
    "035720": "카카오",
    "051910": "LG화학",
    "207940": "삼성바이오로직스",
    "005380": "현대차",
    "000270": "기아",
}

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# ② RS Score 계산 (O'Neil: 최근 1분기 2배 가중)
# ============================================================
QUARTER = 63  # 영업일 기준 3개월

def calc_rs_score(prices: pd.Series) -> float:
    """
    4개 분기 수익률 계산 후 최근 분기 2배 가중 합산.
    데이터 부족 시 NaN 반환.
    """
    n = len(prices)
    if n < QUARTER * 4:
        return np.nan

    def qr(start_offset, end_offset):
        return prices.iloc[-end_offset] / prices.iloc[-start_offset] - 1

    try:
        q1 = qr(QUARTER,         1)       # 최근 3개월
        q2 = qr(QUARTER * 2,     QUARTER) # 3~6개월
        q3 = qr(QUARTER * 3, QUARTER * 2) # 6~9개월
        q4 = qr(QUARTER * 4, QUARTER * 3) # 9~12개월
        return q1 * 2 + q2 + q3 + q4
    except (IndexError, ZeroDivisionError):
        return np.nan

# ============================================================
# ③ 전체 KOSPI RS 백분위 계산
# ============================================================
def build_universe_rs(start: str, end: str, sleep: float = 0.05) -> pd.DataFrame:
    """
    KOSPI 전 종목의 RS Score → 백분위 랭킹(1~99) 산출.
    최초 실행은 시간이 걸립니다 (~10~15분).
    캐시 파일(rs_cache_YYYYMMDD.csv)이 있으면 재사용합니다.
    """
    cache_file = f"rs_cache_{end}.csv"
    if os.path.exists(cache_file):
        print(f"[캐시 로드] {cache_file}")
        return pd.read_csv(cache_file, dtype={"Code": str})

    tickers = fdr.StockListing("KOSPI")[["Code", "Name"]]
    records = []

    for idx, row in tickers.iterrows():
        code, name = row["Code"], row["Name"]
        try:
            df = fdr.DataReader(code, start, end)
            if df.empty or len(df) < QUARTER * 4:
                continue
            score = calc_rs_score(df["Close"])
            if not np.isnan(score):
                records.append({"Code": code, "Name": name, "Score": score})
            time.sleep(sleep)
        except Exception as e:
            print(f"  [{code}] 오류: {e}")

    universe = pd.DataFrame(records)
    universe["Rank"] = universe["Score"].rank(ascending=True)
    universe["RS"]   = ((universe["Rank"] - 1) / (len(universe) - 1) * 98 + 1).astype(int)
    universe.to_csv(cache_file, index=False)
    print(f"[완료] {len(universe)}개 종목 RS 계산 → {cache_file}")
    return universe

# ============================================================
# ④ RS Line 기울기 (종목가격 / KOSPI 지수 비율 추세)
# ============================================================
def rs_line_slope(prices: pd.Series, kospi: pd.Series, window: int = 20) -> float:
    """최근 window일 RS Line 변화율 (%)"""
    aligned = prices.align(kospi, join="inner")
    rs_line = aligned[0] / aligned[1]
    if len(rs_line) < window + 1:
        return np.nan
    return round((rs_line.iloc[-1] / rs_line.iloc[-window] - 1) * 100, 2)

# ============================================================
# ⑤ Minervini 트렌드 템플릿 통과 여부
# ============================================================
def minervini_check(prices: pd.Series) -> bool:
    if len(prices) < 260:
        return False
    c  = prices.iloc[-1]
    ma50  = prices.tail(50).mean()
    ma150 = prices.tail(150).mean()
    ma200 = prices.tail(200).mean()
    last_month_ma200 = prices.tail(220).head(200).mean()
    hi52  = prices.tail(260).max()
    lo52  = prices.tail(260).min()
    return (
        c > ma50 > ma150 > ma200 > last_month_ma200
        and c > lo52 * 1.30
        and c >= hi52 * 0.75
    )

# ============================================================
# ⑥ 기관 순매수 (pykrx)
# ============================================================
def get_inst_net_buy(date_str: str) -> float:
    """KOSPI 기관합계 순매수 금액 (억 원)"""
    try:
        df = krx.get_market_trading_value_by_date(date_str, date_str, "KOSPI")
        if df.empty:
            return 0.0
        return df["기관합계"].iloc[-1] / 1e8
    except Exception:
        return 0.0

# ============================================================
# ⑦ Telegram 전송
# ============================================================
def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram 미설정] 콘솔 출력:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    })
    if resp.status_code != 200:
        print(f"[Telegram 오류] {resp.text}")

# ============================================================
# ⑧ 메인
# ============================================================
def main():
    today     = dt.date.today()
    end_str   = today.strftime("%Y%m%d")
    start_str = (today - dt.timedelta(days=420)).strftime("%Y%m%d")

    print(f"=== KOSPI RS 모니터 [{today}] ===")

    # (a) KOSPI 지수 가격
    kospi_df = fdr.DataReader("KS11", start_str, end_str)
    kospi_prices = kospi_df["Close"]

    # (b) 기관 순매수
    inst_net = get_inst_net_buy(end_str)
    inst_icon = "🟢" if inst_net > 0 else "🔴"
    inst_line = f"{inst_icon} 기관 순매수: {inst_net:+.0f}억"

    # (c) 전체 우주 RS 계산 (캐시 활용)
    print("RS 유니버스 계산 중 (캐시 없으면 10~15분 소요)...")
    universe = build_universe_rs(start_str, end_str)
    rs_lookup = dict(zip(universe["Code"], universe["RS"]))

    # (d) 관심종목 분석
    results = []
    for code, name in WATCHLIST.items():
        try:
            df = fdr.DataReader(code, start_str, end_str)
            if df.empty:
                continue
            prices = df["Close"]

            rs_rating = rs_lookup.get(code, None)
            slope     = rs_line_slope(prices, kospi_prices)
            trend_ok  = minervini_check(prices)

            hi52  = prices.tail(260).max()
            pct_from_hi = round((prices.iloc[-1] / hi52 - 1) * 100, 1)

            vol = df["Volume"].iloc[-1]
            vol_avg = df["Volume"].tail(20).mean()
            vol_ratio = round(vol / vol_avg, 2) if vol_avg > 0 else np.nan

            results.append({
                "종목명":       name,
                "RS":          rs_rating,
                "RS라인(20일%)": slope,
                "미너비니":     "✅" if trend_ok else "—",
                "52주고점대비(%)": pct_from_hi,
                "거래량비율":   vol_ratio,
            })
        except Exception as e:
            print(f"  [{name}] 오류: {e}")

    df_result = pd.DataFrame(results).sort_values("RS", ascending=False)
    out_file = f"rs_report_{end_str}.csv"
    df_result.to_csv(out_file, index=False, encoding="utf-8-sig")

    # (e) Telegram 메시지 조합
    lines = [f"📊 *KOSPI RS 모니터 [{today.strftime('%m/%d')}]*", inst_line, ""]
    for _, r in df_result.iterrows():
        rs = r["RS"]
        flag = "🔥" if rs and rs >= 90 else ("✅" if rs and rs >= 80 else "⚪")
        lines.append(
            f"{flag} *{r['종목명']}*  RS={rs}  "
            f"RS라인: {r['RS라인(20일%)'  ]:+.1f}%  "
            f"거래량: {r['거래량비율']}x  {r['미너비니']}"
        )
    send_telegram("\n".join(lines))

    print(df_result.to_string(index=False))
    print(f"\n저장: {out_file}")

if __name__ == "__main__":
    main()
