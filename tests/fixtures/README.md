# 測試 Fixtures

本目錄包含測試所需的固定數據文件。

## 📁 文件列表

### attack_test_cases.json

**用途**: 安全攻擊測試案例集合

**內容結構**:
```json
{
  "test_cases": [
    {
      "category": "攻擊類別",
      "cases": [
        {
          "input": "測試輸入",
          "expected_behavior": "預期行為",
          "severity": "嚴重程度"
        }
      ]
    }
  ],
  "test_metadata": {
    "version": "版本號",
    "last_updated": "更新日期",
    "total_cases": "總案例數"
  }
}
```

**使用方式**:
- 由 `tests/conftest.py` 中的 `attack_test_cases` fixture 載入
- 用於 `tests/test_security_attacks.py` 的參數化測試

**測試類別**:
1. **Prompt Injection** - 提示詞注入攻擊
2. **Role Manipulation** - 角色操控攻擊
3. **Jailbreak** - 越獄攻擊
4. **System Leakage** - 系統資訊洩露
5. **Command Injection** - 命令注入攻擊
6. **Delimiter Manipulation** - 分隔符操控
7. **Length Attacks** - 長度攻擊
8. **Irrelevant Questions** - 無關問題

## 🔧 如何使用

在測試中使用 fixture：

```python
def test_example(attack_test_cases):
    """使用攻擊測試案例的範例。"""
    for category in attack_test_cases.get("test_cases", []):
        for case in category["cases"]:
            input_text = case["input"]
            # 執行測試...
```

## 📝 更新數據

如需更新測試案例：

1. 編輯 `attack_test_cases.json`
2. 確保 JSON 格式正確
3. 運行測試驗證：`uv run pytest tests/test_security_attacks.py -v`

---

**相關文檔**:
- [測試指南](../../docs/TESTING.md)
- [攻擊測試](../../docs/ATTACK_TESTING.md)
