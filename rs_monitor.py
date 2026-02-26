#!/usr/bin/env python3
import os, time, glob
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
    if len(prices) < QUARTER * 4:
        return np.nan
    def qr(s, e):
        return prices.iloc[-e] / prices.iloc[-s] - 1
    try:
        return qr(QUARTER,1)*2 + qr(QUARTER*2,QUARTER) + qr(QUARTER*3,QUARTER*2) + qr(QUARTER*4,QUARTER*3)
    except:
        return np.nan

def build_universe_rs(start, end):
    cache_file = "rs_cache_{}.csv".format(end)
    if os.path.exists(cache_file):
        print("[캐시 로드] " + cache_file)
        return pd.read_csv(cache_file, dtype={"Code": str})
    kospi  = fdr.StockListing("KOSPI")[["Code","Name"]]
    kosdaq = fdr.StockListing("KOSDAQ")[["Code","Name"]]
    etf    = fdr.StockListing("ETF/KR")[["Code","Name"]]
    tickers = pd.concat([kospi, kosdaq, etf], ignore_index=True)
    records = []
    for _, row in tickers.iterrows():
        try:
            df = fdr.DataReader(row["Code"], start, end)
            if df.empty or len(df) < QUARTER*4:
                continue
            score = calc_rs_score(df["Close"])
            if not np.isnan(score):
                records.append({"Code": row["Code"], "Name": row["Name"], "Score": score})
            time.sleep(0.05)
        except:
            pass
    u = pd.DataFrame(records)
    u["Rank"] = u["Score"].rank(ascending=True)
    u["RS"]   = ((u["Rank"]-1)/(len(u)-1)*98+1).astype(int)
    u.to_csv(cache_file, index=False)
    print("[완료] {}개 종목 → {}".format(len(u), cache_file))
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
    ma50, ma150, ma200 = prices.tail(50).mean(), prices.tail(150).mean(), prices.tail(200).mean()
    prev200 = prices.tail(220).head(200).mean()
    hi52, lo52 = prices.tail(260).max(), prices.tail(260).min()
    return (c > ma50 > ma150 > ma200 > prev200 and c > lo52*1.30 and c >= hi52*0.75)

def get_inst_net_buy(date_str):
    try:
        df = krx.get_market_trading_value_by_date(date_str, date_str, "KOSPI")
        return 0.0 if df.empty else df["기관합계"].iloc[-1]/1e8
    except:
        return 0.0

def get_inst_trend(days=5):
    results, today, checked, delta = [], dt.date.today(), 0, 0
    while checked < days:
        d = today - dt.timedelta(days=delta)
        delta += 1
        if d.weekday() >= 5:
            continue
        try:
            df = krx.get_market_trading_value_by_date(d.strftime("%Y%m%d"), d.strftime("%Y%m%d"), "KOSPI")
            if not df.empty:
                results.append((d.strftime("%m/%d"), df["기관합계"].iloc[-1]/1e8))
                checked += 1
        except:
            pass
    return list(reversed(results))

def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(msg)
        return
    requests.post(
        "https://api.telegram.org/bot{}/sendMessage".format(TELEGRAM_TOKEN),
        data={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    )

def run_full():
    today = dt.date.today()
    end_str   = today.strftime("%Y%m%d")
    start_str = (today - dt.timedelta(days=420)).strftime("%Y%m%d")
    kospi_prices = fdr.DataReader("KS11", start_str, end_str)["Close"]
    inst_net  = get_inst_net_buy(end_str)
    universe  = build_universe_rs(start_str, end_str)
    rs_lookup = dict(zip(universe["Code"], universe["RS"]))
    results = []
    for code, name in WATCHLIST.items():
        try:
            df = fdr.DataReader(code, start_str, end_str)
            if df.empty:
                continue
            prices = df["Close"]
            vol_avg = df["Volume"].tail(20).mean()
            results.append({
                "종목명": name,
                "RS": rs_lookup.get(code, None),
                "RS라인(20일%)": rs_line_slope(prices, kospi_prices),
                "미너비니": "V" if minervini_check(prices) else "-",
                "52주고점대비(%)": round((prices.iloc[-1]/prices.tail(260).max()-1)*100, 1),
                "거래량비율": round(df["Volume"].iloc[-1]/vol_avg, 2) if vol_avg > 0 else np.nan,
            })
        except Exception as e:
            print("[{}] 오류: {}".format(name, e))
    df_r = pd.DataFrame(results).sort_values("RS", ascending=False)
    df_r.to_csv("rs_report_{}.csv".format(end_str), index=False, encoding="utf-8-sig")
    icon = "🟢" if inst_net > 0 else "🔴"
    lines = ["📊 *KOSPI RS 마감 리포트 [{}]*".format(today.strftime("%m/%d")),
             "{} 기관 순매수: {:+.0f}억".format(icon, inst_net), ""]
    for _, r in df_r.iterrows():
        rs = r["RS"]
        flag = "🔥" if rs and rs >= 90 else ("✅" if rs and rs >= 80 else "⚪")
        lines.append("{} *{}*  RS={}  RS라인: {:+.1f}%  거래량: {}x  {}".format(
            flag, r["종목명"], rs, r["RS라인(20일%)"], r["거래량비율"], r["미너비니"]))
    send_telegram("\n".join(lines))
    print(df_r.to_string(index=False))

def run_morning():
    today = dt.date.today()
    caches  = sorted(glob.glob("rs_cache_*.csv"), reverse=True)
    reports = sorted(glob.glob("rs_report_*.csv"), reverse=True)
    if not caches or not reports:
        send_telegram("캐시 없음 — 오후 4:30 full 모드를 먼저 실행하세요.")
        return
    cache_date = caches[0].replace("rs_cache_","").replace(".csv","")
    df_r  = pd.read_csv(reports[0])
    trend = get_inst_trend(days=5)
    tlines = ["  {} {}: {:+.0f}억".format("🟢" if v>0 else "🔴", d, v) for d,v in trend]
    lines = [
        "🌅 *장 시작 브리핑 [{}]*".format(today.strftime("%m/%d")),
        "📋 기준: {}.{}.{} 종가".format(cache_date[:4], cache_date[4:6], cache_date[6:]),
        "", "*최근 5일 기관 순매수 추이*",
    ] + tlines + ["", "*관심종목 RS 현황*", ""]
    for _, r in df_r.sort_values("RS", ascending=False).iterrows():
        rs = r["RS"]
        flag = "🔥" if rs >= 90 else ("✅" if rs >= 80 else "⚪")
        lines.append("{} *{}*  RS={}  RS라인: {:+.1f}%  {}".format(
            flag, r["종목명"], rs, r["RS라인(20일%)"], r["미너비니"]))
    send_telegram("\n".join(lines))

def run_intraday():
    today    = dt.date.today()
    inst_net = get_inst_net_buy(today.strftime("%Y%m%d"))
    icon = "🟢" if inst_net > 0 else "🔴"
    if inst_net > 3000:    signal = "💪 강한 유입 — 매매 적극 유지"
    elif inst_net > 0:     signal = "👀 순유입 — 관망 유지"
    elif inst_net > -3000: signal = "⚠️ 소폭 이탈 — 주의"
    else:                  signal = "🚨 강한 이탈 — 포지션 점검 필요"
    now_kst = dt.datetime.utcnow() + dt.timedelta(hours=9)
    lines = [
        "📡 *장 중 기관 수급 체크 [{}]*".format(now_kst.strftime("%m/%d %H:%M")),
        "{} 기관 순매수(당일 누적): *{:+.0f}억*".format(icon, inst_net),
        "→ " + signal, "",
        "_RS 수치는 오후 4:30 마감 리포트를 참고하세요_"
    ]
    send_telegram("\n".join(lines))

def main():
    mode = RUN_MODE.strip().lower()
    print("[실행 모드] " + mode)
    if mode == "morning":    run_morning()
    elif mode == "intraday": run_intraday()
    else:                    run_full()

if __name__ == "__main__":
    main()
