#!/usr/bin/env python3
"""
整理衛教單檔案的格式，添加適當的換行和段落分隔，但不改變內容。
"""
import re
from pathlib import Path


def format_content(content: str) -> str:
    """
    格式化內容，添加適當的換行和段落分隔。
    保持所有原始內容不變。
    """
    lines = content.split('\n')
    
    # 處理 front matter (YAML header)
    front_matter_end = -1
    for i, line in enumerate(lines):
        if line.strip() == '---' and i > 0:
            front_matter_end = i
            break
    
    if front_matter_end == -1:
        front_matter = ''
        body = '\n'.join(lines)
    else:
        front_matter = '\n'.join(lines[:front_matter_end + 1])
        body = '\n'.join(lines[front_matter_end + 1:])
    
    body = body.strip()
    
    # 先處理被錯誤分開的詞組（如 "居家護理" 被分成 "居家" 和 "護理"）
    # 在格式化之前先修復，避免在格式化過程中再次分離
    common_phrases = [
        (r'居家\s*\n\s*護理', '居家護理'),
        (r'照護\s*\n\s*方法', '照護方法'),
        (r'注意\s*\n\s*事項', '注意事項'),
        (r'操作\s*\n\s*步驟', '操作步驟'),
        (r'準備\s*\n\s*用物', '準備用物'),
        (r'諮詢\s*\n\s*電話', '諮詢電話'),
        (r'接受化學藥物治療\s*\n\s*注意事項', '接受化學藥物治療注意事項'),
        (r'接受化學藥物治療\s+注意事項', '接受化學藥物治療注意事項'),
    ]
    for pattern, replacement in common_phrases:
        body = re.sub(pattern, replacement, body, flags=re.IGNORECASE)
    
    # 處理重複的文字（如 "接受化學藥物治療注意事項" 重複多次）
    def remove_repeated_text(text):
        # 先處理明顯的重複模式
        # 找出連續重複的相同文字（至少5個字元，重複至少2次）
        # 處理各種分隔符號的情況
        max_iterations = 15  # 防止無限循環
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            original = text
            
            # 處理空格分隔的重複（如 "接受化學藥物治療注意事項 接受化學藥物治療注意事項"）
            text = re.sub(r'(.{5,}?)(?:\s+\1){1,}', r'\1', text)
            
            # 處理換行分隔的重複（包括多個換行）
            text = re.sub(r'(.{5,}?)(?:\n\s*\1){1,}', r'\1', text)
            
            # 處理緊接的重複（沒有分隔符）
            text = re.sub(r'(.{5,}?)(\1){1,}', r'\1', text)
            
            # 處理特定常見重複詞組
            specific_repeats = [
                (r'接受化學藥物治療注意事項', r'接受化學藥物治療注意事項'),
                (r'居家護理', r'居家護理'),
                (r'照護方法', r'照護方法'),
                (r'常見之副作用與照護', r'常見之副作用與照護'),
            ]
            for pattern, _ in specific_repeats:
                # 移除重複的詞組（至少重複2次）
                text = re.sub(rf'({re.escape(pattern)})(?:\s+\1){{1,}}', r'\1', text)
                text = re.sub(rf'({re.escape(pattern)})(?:\n\s*\1){{1,}}', r'\1', text)
            
            # 如果沒有變化，停止
            if text == original:
                break
        
        return text
    
    body = remove_repeated_text(body)
    
    # 如果內容擠在一起（沒有適當換行），需要智能分段
    body_lines = body.split('\n')
    non_empty_lines = [l for l in body_lines if l.strip()]
    
    # 如果非空行很少但內容很長，說明內容擠在一起
    needs_formatting = len(non_empty_lines) <= 3 and len(body) > 200
    
    if needs_formatting:
        # 需要智能分段
        
        # 1. 處理編號模式（數字編號，但確保後面有空格）
        body = re.sub(r'(\d+\.\s+)', r'\n\n\1', body)
        
        # 2. 處理中文編號（一、二、三等）
        body = re.sub(r'([一二三四五六七八九十]+、\s*)', r'\n\n\1', body)
        
        # 3. 處理括號編號（(1) (2) 等）
        body = re.sub(r'(\([一二三四五六七八九十\d]+\)\s*)', r'\n\n\1', body)
        
        # 4. 處理特殊符號標記（但不要分開 "圖" 和 "出自於"）
        body = re.sub(r'(※\s*)', r'\n\n\1', body)
        # 處理 "圖出自於" 保持在一起
        body = re.sub(r'(圖出自於[^\n]+)', r'\n\n\1', body)
        # 處理單獨的 "圖" 後面跟 "出自於"
        body = re.sub(r'(圖)\s+(出自於)', r'\1\2', body)
        body = re.sub(r'(屏基關心您)', r'\n\n\1', body)
        
        # 5. 處理結構性關鍵字（在關鍵字前加換行）
        structural_keywords = [
            r'目的：', r'適應症：', r'禁忌症：', r'處置流程：', r'處置流程：',
            r'併發症/風險：', r'替代方案：', r'檢查後處置：',
            r'一、', r'二、', r'三、', r'四、', r'五、', r'六、', r'七、', r'八、', r'九、', r'十、',
            r'照護方法：', r'居家護理：', r'注意事項', r'操作步驟：',
            r'準備用物：', r'諮詢電話：', r'症狀：', r'護理：'
        ]
        for keyword in structural_keywords:
            body = re.sub(rf'([^\n])({keyword})', r'\1\n\n\2', body)
        
        # 6. 處理編號和日期信息（保持在同一行）
        # 將 "編 號 ND 內-035 製訂日期 2003.11 九修日期 2023.06" 這樣的內容保持在一行
        body = re.sub(r'(編 號[^\n]+?修日期[^\n]+?)(\n\n)', r'\1 ', body)
        
        # 7. 處理句號後的段落分隔（但不要破壞編號後的內容）
        # 只在句號後且下一段不是編號時才分段
        body = re.sub(r'([。！？])\s*([^\d一二三四五六七八九十\(※圖屏基])', r'\1\n\n\2', body)
        
    else:
        # 已經有換行，只需要清理和優化
        # 移除多餘的空白行
        cleaned_lines = []
        prev_empty = False
        for line in body_lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            cleaned_lines.append(line)
            prev_empty = is_empty
        body = '\n'.join(cleaned_lines)
        
        # 確保結構性關鍵字前有適當的換行
        structural_keywords = [
            r'目的：', r'適應症：', r'禁忌症：', r'處置流程：', r'處置流程：',
            r'併發症/風險：', r'替代方案：', r'檢查後處置：',
            r'一、', r'二、', r'三、', r'四、', r'五、', r'六、', r'七、', r'八、', r'九、', r'十、',
            r'照護方法：', r'居家護理：', r'注意事項', r'操作步驟：',
            r'準備用物：', r'諮詢電話：', r'症狀：', r'護理：'
        ]
        for keyword in structural_keywords:
            body = re.sub(rf'([^\n])({keyword})', r'\1\n\n\2', body)
    
    # 修復被錯誤分開的 "圖" 和 "出自於"
    body = re.sub(r'(圖)\s*\n\s*(出自於)', r'\1\2', body)
    
    # 再次修復被錯誤分開的詞組（在格式化之後）
    common_phrases_final = [
        (r'居家\s*\n\s*護理', '居家護理'),
        (r'照護\s*\n\s*方法', '照護方法'),
        (r'注意\s*\n\s*事項', '注意事項'),
        (r'操作\s*\n\s*步驟', '操作步驟'),
        (r'準備\s*\n\s*用物', '準備用物'),
        (r'諮詢\s*\n\s*電話', '諮詢電話'),
    ]
    for pattern, replacement in common_phrases_final:
        body = re.sub(pattern, replacement, body, flags=re.IGNORECASE)
    
    # 清理多餘的換行（超過兩個連續換行）
    body = re.sub(r'\n{3,}', '\n\n', body)
    
    # 移除開頭和結尾的空白行
    body = body.strip()
    
    # 組合結果
    if front_matter:
        result = front_matter + '\n\n' + body
    else:
        result = body
    
    return result


def process_file(file_path: Path):
    """處理單個檔案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 格式化內容
        formatted_content = format_content(content)
        
        # 寫回檔案
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        print(f"✓ 已處理: {file_path.name}")
        return True
    except (IOError, UnicodeDecodeError, UnicodeEncodeError) as e:
        print(f"✗ 處理失敗 {file_path.name}: {e}")
        return False


def main():
    """主函數"""
    # 衛教單資料夾路徑
    base_dir = Path(__file__).parent.parent
    health_edu_dir = base_dir / 'rag_test_data' / 'docs' / '衛教單'
    
    if not health_edu_dir.exists():
        print(f"錯誤：找不到資料夾 {health_edu_dir}")
        return
    
    # 取得所有 .md 檔案
    md_files = list(health_edu_dir.glob('*.md'))
    
    if not md_files:
        print(f"錯誤：在 {health_edu_dir} 中找不到 .md 檔案")
        return
    
    print(f"找到 {len(md_files)} 個檔案，開始處理...\n")
    
    success_count = 0
    for md_file in sorted(md_files):
        if process_file(md_file):
            success_count += 1
    
    print(f"\n完成！成功處理 {success_count}/{len(md_files)} 個檔案")


if __name__ == '__main__':
    main()

