---
type: guide.location
id: guide-location-<stable-id>
title:
  zh-Hant: <院區與樓層介紹：XX院區>
summary:
  zh-Hant: <提供院區配置、樓層索引、重要動線與服務點>
org:
  name_zh: <醫院名稱>
lang: zh-Hant
tags: [院區, 樓層, 動線, 櫃台, 電梯, 無障礙, <院區>]
audience: [patient, visitor]

location:
  campus_zh: <院區名稱>
  buildings:
    - building_zh: <大樓/棟別>
      floors:
        - floor: B1
          highlights: [<服務點/單位/櫃台/出入口>]
        - floor: 1F
          highlights: [<掛號/批價/急診入口…>]
  wayfinding:
    - from: <入口A>
      to: <目標（例如：影像醫學科）>
      steps:
        - <步驟1>
        - <步驟2>

assets:
  images:
    - path: assets/location/<floorplan>.png
      alt: <樓層平面圖替代文字，包含關鍵位置>
      caption: <例如：1F 平面圖—掛號/批價在東側，電梯在中間>
      role: floorplan

retrieval:
  aliases: [<院區俗稱>, <棟別俗稱>]
  keywords: [樓層, 平面圖, 電梯, 櫃台, 門診, 檢查室]

source:
  url: <來源網址>
  captured_at: <YYYY-MM-DD>
updated_at: <YYYY-MM-DD>
---

# <院區與樓層介紹：XX院區>

## 概要
...

## 樓層索引
### <棟別A>
- B1：...
- 1F：...
- 2F：...

## 重要動線（怎麼走）
- 從 <入口A> 到 <目標>：
  1. ...
  2. ...

## 圖示（樓層/導引）
![<alt>](assets/location/<floorplan>.png "1F 平面圖")
- 圖說：<caption>

## 聯絡/承辦窗口
- ...
