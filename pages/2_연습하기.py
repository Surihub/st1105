from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="펭귄 데이터 대시보드", page_icon="🐧", layout="wide")
MEASURES = {
    "bill_length_mm": "부리 길이 (mm)",
    "bill_depth_mm": "부리 깊이 (mm)",
    "flipper_length_mm": "날개 길이 (mm)",
    "body_mass_g": "몸무게 (g)",
}
LABELS = {"species": "종", "island": "서식 섬", "sex": "성별", **MEASURES}


@st.cache_data(show_spinner="펭귄 데이터를 불러오는 중입니다…")
def load_data():
    local = Path(__file__).resolve().parents[1] / "data" / "penguins.csv"
    source = local if local.exists() else "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    return pd.read_csv(source)


st.title("🐧 펭귄 데이터 대시보드")
st.caption("Palmer Penguins · 남극 펭귄의 종과 서식지, 신체 특징을 탐색해 보세요.")
with st.sidebar:
    st.header("데이터와 필터")
    uploaded = st.file_uploader("내 펭귄 CSV 불러오기 (선택)", type="csv")

try:
    data = pd.read_csv(uploaded) if uploaded is not None else load_data()
except Exception:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결을 확인하거나 사이드바에서 펭귄 CSV를 업로드해 주세요.")
    st.info("로컬 데이터는 프로젝트의 data/penguins.csv에 저장하면 자동으로 불러옵니다.")
    st.stop()

missing = set(LABELS) - set(data.columns)
if missing:
    st.error("CSV에 필요한 열이 없습니다: " + ", ".join(sorted(missing)))
    st.stop()
data = data[list(LABELS)].copy()
for column in MEASURES:
    data[column] = pd.to_numeric(data[column], errors="coerce")
for column in ("species", "island", "sex"):
    data[column] = data[column].fillna("미상").astype(str)

filtered = data.copy()
with st.sidebar:
    for column in ("species", "island", "sex"):
        options = sorted(data[column].unique())
        selected = st.multiselect(LABELS[column], options, default=options)
        filtered = filtered[filtered[column].isin(selected)]
    st.caption("미상은 분류 정보가 없는 관측값입니다.")

st.caption(f"전체 {len(data):,}마리 중 {len(filtered):,}마리 선택됨 · 모든 지표와 차트에 필터가 적용됩니다.")
if filtered.empty:
    st.info("조건에 맞는 펭귄이 없습니다. 사이드바에서 필터를 다시 선택해 주세요.")
    st.stop()

count, species, weight, flipper = st.columns(4)
count.metric("선택한 펭귄", f"{len(filtered):,}마리")
species.metric("펭귄 종", f"{filtered.species.nunique()}종")
for tile, column, label, unit in (
    (weight, "body_mass_g", "평균 몸무게", "g"),
    (flipper, "flipper_length_mm", "평균 날개 길이", "mm"),
):
    value = filtered[column].mean()
    tile.metric(label, "정보 없음" if pd.isna(value) else f"{value:,.1f} {unit}")

overview, compare, table = st.tabs(["📊 분포와 관계", "📐 종별 비교", "📋 데이터 보기"])
with overview:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("신체 특징 사이의 관계")
        xcol, ycol = st.columns(2)
        x = xcol.selectbox("가로축", list(MEASURES), index=2, format_func=MEASURES.get)
        y = ycol.selectbox("세로축", list(MEASURES), index=3, format_func=MEASURES.get)
        points = filtered.dropna(subset=[x, y])
        if points.empty:
            st.info("선택한 항목에 유효한 관측값이 없습니다.")
        else:
            st.scatter_chart(points, x=x, y=y, color="species", x_label=MEASURES[x], y_label=MEASURES[y], height=380)
        st.caption(f"유효한 관측값 {len(points):,}개 · 선택한 축에 결측값이 있는 행은 제외됩니다.")
    with right:
        st.subheader("종별 개체 수")
        st.bar_chart(filtered.groupby("species").size().rename("개체 수").rename_axis("종"), color="#36a6a0", height=380)
    st.subheader("섬별 서식 분포")
    st.bar_chart(pd.crosstab(filtered.island, filtered.species).rename_axis("서식 섬"), stack=True, height=280)

with compare:
    st.subheader("종별 평균 신체 크기")
    measure = st.selectbox("비교할 항목", list(MEASURES), index=3, format_func=MEASURES.get)
    means = filtered.groupby("species")[measure].mean().dropna()
    if means.empty:
        st.info("이 항목에 유효한 관측값이 없습니다.")
    else:
        st.bar_chart(means.rename(MEASURES[measure]).rename_axis("종"), color="#36a6a0")
    st.caption("평균과 표준편차는 각 항목의 결측값을 제외하고 계산합니다. 항목마다 표본 수가 다를 수 있습니다.")
    summary = filtered.groupby("species")[list(MEASURES)].agg(["count", "mean", "std"])
    stats = {"count": "표본 수", "mean": "평균", "std": "표준편차"}
    summary.columns = [f"{MEASURES[col]} · {stats[stat]}" for col, stat in summary.columns]
    st.dataframe(summary.round(2).rename_axis("종"), use_container_width=True)

with table:
    st.subheader("선택한 펭귄 데이터")
    st.dataframe(filtered.rename(columns=LABELS), use_container_width=True, hide_index=True)
    st.download_button("선택한 데이터 CSV 다운로드", filtered.to_csv(index=False).encode("utf-8-sig"), file_name="penguins_filtered.csv", mime="text/csv")
    with st.expander("결측값 확인"):
        st.caption("분류 정보의 결측값은 ‘미상’으로 표시하며, 신체 측정값의 결측값은 임의로 채우지 않습니다.")
        st.dataframe(filtered[list(MEASURES)].isna().sum().rename(index=MEASURES).rename("결측값 수"))

st.divider()
st.caption("출처: Palmer Penguins / seaborn-data · 길이는 mm, 몸무게는 g 단위입니다.")
st.markdown("[데이터 원본 보기](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)")
