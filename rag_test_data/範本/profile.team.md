---
type: hospital_team
id: team-<stable-id>

name:
  zh-Hant: <團隊/科別名稱>
  en: <English name (optional)>

org:
  name_zh: <醫院名稱>

# 例：內科、外科、兒科...
parent_department: <上層部門/科系>

# 可多個；用 primary 標主專長/主頁籤
subspecialties:
  - name_zh: <次專科/分科A>
    primary: true
  - name_zh: <次專科/分科B>
    primary: false

availability:
  outpatient: true
  inpatient: true
  emergency_24h: false

locations:
  - name_zh: <院區/樓層/單位（可先粗略）>

# 服務範圍：建議至少維持其中 2 類 list，未提供就留空陣列
service_scope:
  conditions:      # 疾病/適應症
    - <疾病/問題1>
    - <疾病/問題2>
  services:        # 服務型（門診/衛教/照護計畫）
    - <服務1>
  tests:           # 檢查
    - <檢查1>
  procedures:      # 處置/治療/手術/介入
    - <治療1>
  programs:        # 專案/中心/快速通道
    - <計畫1>

equipment:
  - <設備/儀器1>
  - <設備/儀器2>

staffing:
  physicians_count: null
  nurses_count: null
  technologists_count: null
  notes:
    - <人力/分工補充（可選）>

highlights:
  - <特色1>
  - <特色2>

metrics:
  - label: <量化項目名稱>
    value: <數值/敘述>
    period: <期間（例如每月/每年）>

retrieval:
  aliases:
    - <別名/常見寫法1>
    - <別名/常見寫法2>
  keywords:
    - <關鍵詞1>
    - <關鍵詞2>

source:
  url: <來源網址>
  captured_at: <YYYY-MM-DD>
updated_at: <YYYY-MM-DD>
---

# <團隊/科別名稱>

## 團隊簡介
<用 3–6 句話描述：成立背景、定位、主要服務對象、跨科合作等。>

## 團隊編制
- 醫師：<人數或描述>
- 護理/照護：<描述>
- 醫事技術/其他：<描述>
- 特色分工：<例如：衛教師、個管師、呼吸治療師等>

## 服務與治療
### 服務範圍（疾病/問題）
- <疾病/問題1>
- <疾病/問題2>

### 主要治療/處置
- <治療1>：<一句話說明適用情境或特色>
- <治療2>：...

### 特色門診/照護計畫（若有）
- <計畫1>：<一句話說明>

## 檢查項目
- <檢查1>
- <檢查2>

## 設備與特色
- <設備/儀器1>：<用途/亮點>
- <設備/儀器2>：...

## 品質與安全
- <品質/流程/病安亮點（例如：24 小時服務、快速通道、跨科討論等）>
- <若有認證/指標/教育訓練也可列點>

## 就醫資訊
- 門診：<是否需轉診/掛號方式（可留空）>
- 住院：<收治範圍/流程（可留空）>
- 急診/24 小時：<如適用，寫清楚>
- 轉介：<院內/院外轉介說明（可留空）>

## 資料來源
- <來源網址>（擷取日期：<YYYY-MM-DD>）
