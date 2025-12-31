#!/usr/bin/env python3
"""
下載屏基醫院樓層指引圖片
"""
import os
import requests
from urllib.parse import unquote
from pathlib import Path

# 圖片 URL 列表
image_urls = [
    # 各大樓分佈平面圖
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E5%90%84%E5%A4%A7%E6%A8%93%E5%88%86%E4%BD%88%E5%B9%B3%E9%9D%A2%E5%9C%96.jpg",
    # 路加大樓
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A0B2.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A0B1.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A01F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A02Fn.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A03Fn.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A04F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A05F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A06F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E8%B7%AF%E5%8A%A07F.jpg",
    # 馬太大樓
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%93B1.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%931F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%932F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%933F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%934F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%935F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%936F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%A4%AA%E5%A4%A7%E6%A8%937F-1.jpg",
    # 馬可大樓
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%8F%AF%E5%A4%A7%E6%A8%931f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%8F%AF%E5%A4%A7%E6%A8%932f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%8F%AF%E5%A4%A7%E6%A8%933f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%8F%AF%E5%A4%A7%E6%A8%934f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%8F%AF%E5%A4%A7%E6%A8%935f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%A6%AC%E5%8F%AF%E5%A4%A7%E6%A8%936f.jpg",
    # 韓偉大樓
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%9F%93%E5%81%89%E5%A4%A7%E6%A8%93b1.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%9F%93%E5%81%89%E5%A4%A7%E6%A8%931f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%9F%93%E5%81%89%E5%A4%A7%E6%A8%932f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%9F%93%E5%81%89%E5%A4%A7%E6%A8%933f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%9F%93%E5%81%89%E5%A4%A7%E6%A8%934f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%9F%93%E5%81%89%E5%A4%A7%E6%A8%935f.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E9%9F%93%E5%81%89%E5%A4%A7%E6%A8%936f.jpg",
    # 約翰大樓
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%93B2.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%93B1.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%931F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%932F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%933F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%934F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%935F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E7%B4%84%E7%BF%B0%E5%A4%A7%E6%A8%936F.jpg",
    # 恩慈大樓
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E6%81%A9%E6%85%88%E5%A4%A7%E6%A8%93B2.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E6%81%A9%E6%85%88%E5%A4%A7%E6%A8%93B1.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E6%81%A9%E6%85%88%E5%A4%A7%E6%A8%931F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E6%81%A9%E6%85%88%E5%A4%A7%E6%A8%932F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E6%81%A9%E6%85%88%E5%A4%A7%E6%A8%933F.jpg",
    "https://www.ptch.org.tw/tw/ckfinder/userfiles/images/%E6%81%A9%E6%85%88%E5%A4%A7%E6%A8%934F.jpg",
]

def download_image(url, output_dir):
    """下載圖片到指定目錄"""
    try:
        # 從 URL 提取檔名
        filename = unquote(url.split('/')[-1])
        output_path = output_dir / filename
        
        # 如果檔案已存在，跳過
        if output_path.exists():
            print(f"已存在: {filename}")
            return output_path
        
        # 下載圖片
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # 儲存圖片
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"下載成功: {filename}")
        return output_path
    except Exception as e:
        print(f"下載失敗 {url}: {e}")
        return None

def main():
    # 設定輸出目錄
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    images_dir = project_root / "rag_test_data" / "cus" / "醫院簡介" / "關於我們" / "樓層指引圖片"
    
    # 建立目錄
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"圖片將下載到: {images_dir}")
    print(f"共 {len(image_urls)} 張圖片\n")
    
    # 下載所有圖片
    downloaded = []
    for url in image_urls:
        path = download_image(url, images_dir)
        if path:
            downloaded.append(path)
    
    print(f"\n完成！成功下載 {len(downloaded)} 張圖片")

if __name__ == "__main__":
    main()

