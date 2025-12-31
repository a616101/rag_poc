#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析衛教單.txt，根據範本格式創建獨立的 markdown 檔案
"""

import re
import os
from pathlib import Path
from datetime import datetime

def parse_handout_text(text):
    """解析單個衛教內容，提取元數據和內容"""
    result = {
        'title': '',
        'code': '',
        'created_at': '',
        'revision_date': '',
        'revision_note': '',
        'department': '',
        'content': text.strip()
    }
    
    # 清理開頭的空行和空白
    text = text.strip()
    
    # 先提取編號（這樣可以幫助確定標題的邊界）
    # 編號格式通常是：ND 急-005 或 ND 內-035 等，可能包含中文字符
    code_match = re.search(r'編\s*號\s+(.*?)(?=\s+製訂日期)', text)
    if code_match:
        result['code'] = code_match.group(1).strip()
    else:
        # 嘗試更寬鬆的模式（沒有「製訂日期」的情況）
        code_match = re.search(r'編\s*號\s+([^\s]+(?:\s+[^\s]+)*)', text)
        if code_match:
            result['code'] = code_match.group(1).strip()
    
    # 提取標題（通常在開頭，到「編 號」之前）
    # 嘗試多種模式
    title_match = re.search(r'^(.+?)\s+編\s*號', text, re.MULTILINE)
    if title_match:
        result['title'] = title_match.group(1).strip()
    else:
        # 如果沒有找到「編 號」，嘗試找「編號」（沒有空格）
        title_match = re.search(r'^(.+?)\s*編號\s*', text, re.MULTILINE)
        if title_match:
            result['title'] = title_match.group(1).strip()
        else:
            # 嘗試找「製訂日期」前的內容作為標題
            title_match = re.search(r'^(.+?)\s+製訂日期', text, re.MULTILINE)
            if title_match:
                result['title'] = title_match.group(1).strip()
            else:
                # 嘗試從開頭提取標題，標題通常在「一、」或「壹、」之前
                # 或者以「須知」、「指導」、「注意事項」等結尾
                title_patterns = [
                    r'^(.+?)(?:\s+一、|\s+壹、|\s+1\.|\n一、|\n壹、)',
                    r'^(.+?(?:須知|指導|注意事項|照護|衛教|說明|簡介))',
                    r'^(.{2,50}?)(?:\s+編\s*號|$)',
                ]
                for pattern in title_patterns:
                    title_match = re.search(pattern, text, re.MULTILINE)
                    if title_match:
                        candidate = title_match.group(1).strip()
                        # 驗證標題長度（標題通常不會太長）
                        if 2 <= len(candidate) <= 50:
                            result['title'] = candidate
                            break
                
                # 最後嘗試：取第一行非空內容作為標題（限制長度）
                if not result['title']:
                    first_line = text.split('\n')[0].strip()
                    # 如果第一行太長，嘗試截取到第一個標點符號或「一、」之前
                    if len(first_line) > 50:
                        # 嘗試截取到「一、」或「壹、」之前
                        match = re.search(r'^(.+?)(?:\s+一、|\s+壹、)', first_line)
                        if match:
                            first_line = match.group(1).strip()
                        # 如果還是太長，截取前50個字
                        if len(first_line) > 50:
                            first_line = first_line[:50]
                    if first_line and 2 <= len(first_line) <= 50:
                        result['title'] = first_line
    
    # 如果還沒有找到編號，嘗試在內容最後找編號（有些格式是標題在前，編號在後）
    if not result['code']:
        code_match = re.search(r'編\s*號\s+([A-Z0-9\s\-]+)', text)
        if code_match:
            result['code'] = code_match.group(1).strip()
    
    # 提取製訂日期
    created_match = re.search(r'製訂日期\s+(\d{4}[\./]\d{1,2})', text)
    if created_match:
        date_str = created_match.group(1).replace('/', '.')
        result['created_at'] = date_str
    
    # 提取修訂日期和修訂次數
    revision_match = re.search(r'([一二三四五六七八九十]+)修日期\s+(\d{4}[\./]\d{1,2})', text)
    if revision_match:
        result['revision_note'] = revision_match.group(1) + '修'
        date_str = revision_match.group(2).replace('/', '.')
        result['revision_date'] = date_str
    
    # 從編號推斷科別
    if result['code']:
        if '急' in result['code']:
            result['department'] = '急診科'
        elif '內' in result['code']:
            result['department'] = '內科'
        elif '外' in result['code']:
            result['department'] = '外科'
        elif '心' in result['code'] or '心血管' in result['code']:
            result['department'] = '心臟內科'
        elif '兒' in result['code']:
            result['department'] = '兒科'
        elif '婦' in result['code']:
            result['department'] = '婦產科'
        elif '寧' in result['code']:
            result['department'] = '安寧病房'
        else:
            result['department'] = '其他'
    else:
        # 如果沒有編號，嘗試從標題推斷
        if result['title']:
            if '急診' in result['title'] or '返家' in result['title']:
                result['department'] = '急診科'
            elif '兒科' in result['title'] or '兒童' in result['title'] or '新生兒' in result['title']:
                result['department'] = '兒科'
            elif '婦產' in result['title'] or '產科' in result['title'] or '生產' in result['title']:
                result['department'] = '婦產科'
            elif '外科' in result['title'] or '手術' in result['title']:
                result['department'] = '外科'
            elif '心臟' in result['title'] or '心血管' in result['title']:
                result['department'] = '心臟內科'
            elif '內科' in result['title'] or '內視鏡' in result['title']:
                result['department'] = '內科'
    
    return result

def extract_key_info(content):
    """從內容中提取關鍵資訊"""
    info = {
        'conditions': [],
        'key_actions': [],
        'red_flags': [],
        'keywords': []
    }
    
    # 提取疾病/主題（從標題或內容中）
    # 這裡可以根據實際內容進一步優化
    
    # 提取紅旗症狀（通常包含「立即就醫」、「儘快就醫」等）
    red_flag_patterns = [
        r'如有下列.*?請.*?就醫[：:](.*?)(?:\n|$)',
        r'下列.*?請.*?就醫[：:](.*?)(?:\n|$)',
        r'立即.*?就醫[：:](.*?)(?:\n|$)',
    ]
    
    for pattern in red_flag_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # 分割成多個項目
            items = re.split(r'[。\n\d\.、]', match)
            for item in items:
                item = item.strip()
                if item and len(item) > 3:
                    info['red_flags'].append(item)
    
    return info

def create_handout_file(parsed_data, filename_base, output_dir):
    """根據範本創建 markdown 檔案"""
    
    # 生成穩定 ID（使用檔名）
    stable_id = filename_base.replace('.txt', '').replace('_', '-')
    
    # 提取關鍵資訊
    key_info = extract_key_info(parsed_data['content'])
    
    # 生成檔案內容
    content_lines = [
        "---",
        "type: education.handout",
        f"id: edu-{stable_id}",
        "title:",
        f"  zh-Hant: {parsed_data['title']}",
        "summary:",
        f"  zh-Hant: 提供{parsed_data['title']}相關的衛教資訊",
        "org:",
        "  name_zh: 屏東基督教醫院",
        "lang: zh-Hant",
        "audience: [patient, family]",
        "tags: [衛教",
    ]
    
    # 添加科別標籤
    if parsed_data['department']:
        content_lines.append(f", {parsed_data['department']}")
    
    # 添加疾病/主題標籤（從標題推斷）
    if parsed_data['title']:
        # 簡單提取關鍵詞
        title_keywords = re.findall(r'[^病病人人者須知指導]+', parsed_data['title'])
        if title_keywords:
            main_keyword = title_keywords[0].strip()
            if main_keyword:
                content_lines.append(f", {main_keyword}")
    
    content_lines.append("]")
    content_lines.append("")
    content_lines.append("education:")
    
    # 判斷類別
    category = "discharge"
    if "檢查" in parsed_data['title'] or "檢查須知" in parsed_data['title']:
        if "前" in parsed_data['title']:
            category = "pre_exam"
        else:
            category = "post_exam"
    elif "手術" in parsed_data['title'] or "術後" in parsed_data['title']:
        if "前" in parsed_data['title']:
            category = "pre_op"
        else:
            category = "post_op"
    elif "用藥" in parsed_data['title'] or "藥物" in parsed_data['title']:
        category = "medication"
    elif "飲食" in parsed_data['title'] or "營養" in parsed_data['title']:
        category = "self_care"
    
    content_lines.append(f"  category: {category}")
    content_lines.append(f"  code: {parsed_data['code']}")
    
    if parsed_data['created_at']:
        # 轉換日期格式
        date_parts = parsed_data['created_at'].split('.')
        if len(date_parts) == 2:
            content_lines.append(f"  created_at: {date_parts[0]}-{date_parts[1].zfill(2)}")
    
    if parsed_data['revision_date']:
        content_lines.append("  revisions:")
        date_parts = parsed_data['revision_date'].split('.')
        if len(date_parts) == 2:
            content_lines.append(f"    - date: {date_parts[0]}-{date_parts[1].zfill(2)}")
            content_lines.append(f"      note: {parsed_data['revision_note']}")
    
    content_lines.append("  owner:")
    content_lines.append(f"    department_zh: {parsed_data['department']}")
    
    # 添加疾病/主題
    if parsed_data['title']:
        # 從標題提取主要疾病/主題
        title_clean = re.sub(r'[病人人者須知指導注意事項]', '', parsed_data['title'])
        if title_clean:
            content_lines.append("  conditions:")
            content_lines.append(f"    - {title_clean.strip()}")
    
    # 添加關鍵行動（從內容中提取）
    if key_info['key_actions']:
        content_lines.append("  key_actions:")
        for action in key_info['key_actions'][:3]:  # 最多3個
            content_lines.append(f"    - {action}")
    
    # 添加紅旗症狀
    if key_info['red_flags']:
        content_lines.append("  red_flags:")
        for flag in key_info['red_flags'][:5]:  # 最多5個
            content_lines.append(f"    - {flag}")
    
    # 添加追蹤資訊
    content_lines.append("  followup:")
    content_lines.append(f"    department_zh: {parsed_data['department']}")
    content_lines.append("    note: 請定期門診追蹤")
    
    content_lines.append("")
    content_lines.append("retrieval:")
    content_lines.append("  aliases:")
    content_lines.append(f"    - {parsed_data['title']}")
    
    content_lines.append("  keywords:")
    # 從標題提取關鍵詞
    if parsed_data['title']:
        keywords = re.findall(r'[\u4e00-\u9fff]+', parsed_data['title'])
        for kw in keywords[:5]:
            if len(kw) > 1:
                content_lines.append(f"    - {kw}")
    
    content_lines.append("")
    content_lines.append("source:")
    content_lines.append("  url: 內部衛教資料")
    content_lines.append(f"  captured_at: {datetime.now().strftime('%Y-%m-%d')}")
    content_lines.append(f"updated_at: {datetime.now().strftime('%Y-%m-%d')}")
    content_lines.append(f"last_reviewed: {datetime.now().strftime('%Y-%m-%d')}")
    content_lines.append("---")
    content_lines.append("")
    content_lines.append(f"# {parsed_data['title']}")
    content_lines.append("")
    
    # 添加內容（需要格式化）
    # 這裡可以進一步處理內容，但先保持原樣
    content_lines.append(parsed_data['content'])
    
    # 寫入檔案
    # 使用編號作為檔名的一部分，避免檔名過長
    if parsed_data['code']:
        safe_code = parsed_data['code'].replace(' ', '-').replace('/', '-')
        output_filename = f"{safe_code}_{parsed_data['title']}.md"
    else:
        output_filename = f"{parsed_data['title']}.md"
    
    # 清理檔名中的特殊字元，並限制長度
    output_filename = re.sub(r'[<>:"/\\|?*]', '_', output_filename)
    # 限制檔名長度（保留副檔名）
    if len(output_filename) > 200:
        name_part = output_filename[:-3]  # 去掉 .md
        output_filename = name_part[:197] + ".md"
    
    output_path = output_dir / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content_lines))
    
    return output_path

def main():
    # 設定路徑
    source_file = Path(__file__).parent.parent / "rag_test_data" / "source" / "衛教單.txt"
    output_dir = Path(__file__).parent.parent / "rag_test_data" / "cus" / "衛教"
    
    # 確保輸出目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 讀取源檔案
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割成多個衛教資訊
    # 每個衛教資訊以檔名開頭（格式：數字_數字.txt），可能後面有空行
    pattern = r'(\d+_\d+\.txt)\n\s*\n?(.*?)(?=\n\d+_\d+\.txt\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    # 如果沒有匹配到，嘗試更簡單的模式（沒有空行）
    if not matches:
        pattern = r'(\d+_\d+\.txt)\n(.*?)(?=\n\d+_\d+\.txt\n|$)'
        matches = re.findall(pattern, content, re.DOTALL)
    
    print(f"找到 {len(matches)} 個衛教資訊")
    
    created_files = []
    for filename, handout_content in matches:
        try:
            parsed = parse_handout_text(handout_content)
            if parsed['title']:
                output_path = create_handout_file(parsed, filename, output_dir)
                created_files.append(output_path)
                print(f"已創建: {output_path.name}")
            else:
                print(f"警告: 無法解析標題，跳過 {filename}")
        except Exception as e:
            print(f"錯誤: 處理 {filename} 時發生錯誤: {e}")
    
    print(f"\n總共創建了 {len(created_files)} 個檔案")

if __name__ == "__main__":
    main()

