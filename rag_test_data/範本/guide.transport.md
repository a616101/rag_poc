---
type: guide.transport
id: guide-transport-<stable-id>
title:
  zh-Hant: <交通指引：到XX院區>
summary:
  zh-Hant: <包含公共運輸/開車停車/步行入口的到院方式>
org:
  name_zh: <醫院名稱>
lang: zh-Hant
tags: [交通, 停車, 公車, 捷運, 院區導引, <院區>]
audience: [patient, visitor]

transport:
  destination:
    campus_zh: <院區名稱>
    address_zh: <地址（可選）>
  modes:
    - mode: mrt
      details: <捷運路線/站名/出口/轉乘>
    - mode: bus
      details: <公車路線/站牌/步行時間>
    - mode: car
      details: <導航建議/入口/下客點>
    - mode: taxi
      details: <下車點>
  parking:
    notes:
      - <停車場位置/收費/高度限制/無障礙車位>
  landmarks:
    - <地標/入口辨識點>

assets:
  images:
    - path: assets/transport/<file>.png
      alt: <地圖/路線說明的替代文字>
      caption: <圖說（含站名/出口/入口關鍵詞）>
      role: map

retrieval:
  aliases: [<俗稱>, <院區別名>]
  keywords: [停車場, 入口, 下客區, 捷運, 公車]

source:
  url: <來源網址>
  captured_at: <YYYY-MM-DD>
updated_at: <YYYY-MM-DD>
---

# <交通指引：到XX院區>

## 概要
...

## 大眾運輸
### 捷運
- ...

### 公車
- ...

## 自行開車與停車
- ...

## 到院後怎麼走（入口/櫃台）
- ...

## 圖示
![<alt>](assets/transport/<file>.png "<optional title>")
- 圖說：<caption>

## 聯絡/承辦窗口
- ...
