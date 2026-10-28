// Options must match the data values character for character — analysis matches on the string.
export const AGE_OPTIONS = ["10대", "20대", "30대", "40대", "50대", "60대", "70대 이상"];
export const ATTEND_OPTIONS = ["예", "아니오"];

export const SCALES: Array<{ key: "grounded" | "coherent" | "plausible" | "fluent"; label: string; hint: string; levels: string[] }> = [
  {
    key: "grounded",
    label: "앵커 부합",
    hint: "앵커와 어긋나지 않는가",
    levels: ["모순 두 곳 이상", "명백한 모순 한 곳", "사소한 불일치 한 곳", "애매한 표현 한 곳", "어긋남 전혀 없음"],
  },
  {
    key: "coherent",
    label: "일관성",
    hint: "한 사람의 이야기로 읽히는가",
    levels: ["한 사람으로 안 읽힘", "여러 항목이 다른 사람 같음", "두세 곳이 안 맞음", "한 곳이 조금 겉돎", "자연스럽게 이어짐"],
  },
  {
    key: "plausible",
    label: "개연성",
    hint: "실제로 있을 법한가 · 고정관념 금지",
    levels: ["거의 있을 수 없음", "있을 법하지 않음", "드물지만 가능", "그럴듯함", "매우 그럴듯함"],
  },
  {
    key: "fluent",
    label: "한국어 자연스러움",
    hint: "내용 말고 문장만 본다",
    levels: ["읽기 어려움·외국어 섞임", "번역투·기계적 문장 많음", "어색한 곳 여러 곳", "어색한 곳 하나", "원어민이 쓴 글"],
  },
];

/** Anchor layout: two groups side by side. Short display names (same meaning — data values are shown as stored). */
export const ANCHOR_GROUPS: Array<{ title: string; labels: string[] }> = [
  { title: "인적 사항", labels: ["성별", "연령대", "생활권", "최종학력", "직업", "월 가구소득", "서울 거주기간", "가구원 수 구간"] },
  { title: "여가 · 관람", labels: ["여가시간 충분도 (3단)", "여가 동반자", "오프라인 관람 여부", "연간 관람 횟수", "연간 본인 지불 관람비", "관람 동반자"] },
];
export const ANCHOR_SHORT: Record<string, string> = {
  "월 가구소득": "가구소득(월)",
  "서울 거주기간": "서울 거주",
  "가구원 수 구간": "가구원 수",
  "여가시간 충분도 (3단)": "여가시간",
  "오프라인 관람 여부": "오프라인 관람",
  "연간 관람 횟수": "관람 횟수(년)",
  "연간 본인 지불 관람비": "본인 지불(년)",
};
export const GENRES = [
  "극장 영화",
  "미술관·전시회",
  "박물관",
  "대중공연",
  "연극(뮤지컬 포함)",
  "서양 고전 음악공연",
  "전통예술공연",
  "무용공연",
  "문학행사",
  "야외문화행사",
];
