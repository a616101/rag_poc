# 測試指南

## 測試架構

本專案使用 pytest 進行測試，包含以下測試類型：

### 測試文件

| 文件 | 描述 | 測試數 |
|------|------|--------|
| `test_api.py` | 基礎 API 端點測試 | ~4 |
| `test_input_guard.py` | 輸入驗證和安全防護測試 | ~15 |
| `test_security_attacks.py` | 完整攻擊測試套件（使用 attack_test_cases.json） | ~60+ |
| `conftest.py` | 共享的 fixtures 配置 | - |

## 快速開始

### 運行所有測試

```bash
# 使用 uv
uv run pytest

# 使用 Docker
docker compose --profile test run --rm test
```

### 運行特定測試文件

```bash
# 只運行 API 測試
uv run pytest tests/test_api.py

# 只運行輸入驗證測試
uv run pytest tests/test_input_guard.py

# 只運行完整攻擊測試
uv run pytest tests/test_security_attacks.py
```

## 使用 Markers 運行特定測試

### 可用的 Markers

- `security` - 所有安全相關測試
- `injection` - 提示詞注入測試
- `jailbreak` - 越獄攻擊測試
- `leak` - 系統洩露測試
- `delimiter` - 分隔符操控測試
- `legitimate` - 合法問題測試
- `comprehensive` - 完整測試套件
- `slow` - 耗時較長的測試

### 運行特定標記的測試

```bash
# 運行所有安全測試
uv run pytest -m security

# 運行提示詞注入測試
uv run pytest -m injection

# 運行越獄攻擊測試
uv run pytest -m jailbreak

# 運行合法問題測試（確保沒有誤判）
uv run pytest -m legitimate

# 運行完整攻擊測試套件
uv run pytest -m comprehensive

# 排除耗時測試
uv run pytest -m "not slow"

# 組合 markers（安全測試但排除完整套件）
uv run pytest -m "security and not comprehensive"
```

## 詳細輸出和報告

### 顯示詳細輸出

```bash
# 顯示詳細測試輸出
uv run pytest -v

# 顯示每個測試的輸出（包括 print）
uv run pytest -v -s

# 顯示測試覆蓋率
uv run pytest --cov=chatbot_rag --cov-report=html
```

### 只運行失敗的測試

```bash
# 首次運行
uv run pytest

# 只重新運行失敗的測試
uv run pytest --lf

# 先運行失敗的，再運行其他
uv run pytest --ff
```

## 測試場景

### 1. 基礎 API 測試 (`test_api.py`)

測試基本 API 端點：
- ✅ Root endpoint
- ✅ API v1 root
- ✅ Health check
- ✅ Hello endpoint

```bash
uv run pytest tests/test_api.py -v
```

### 2. 輸入驗證測試 (`test_input_guard.py`)

測試輸入驗證功能：
- ✅ 合法問題處理
- ✅ 不相關問題拒絕
- ✅ 長度限制
- ✅ 提示詞注入阻止
- ✅ 角色操控阻止
- ✅ 越獄攻擊阻止
- ✅ 系統洩露防止
- ✅ 分隔符操控檢測

```bash
# 運行所有輸入驗證測試
uv run pytest tests/test_input_guard.py -v

# 只運行注入測試
uv run pytest tests/test_input_guard.py::TestPromptInjection -v
```

### 3. 完整攻擊測試 (`test_security_attacks.py`)

使用 `attack_test_cases.json` 中的 60+ 個測試案例：

```bash
# 運行所有攻擊測試（按類別分組）
uv run pytest tests/test_security_attacks.py -v

# 運行特定類別
uv run pytest tests/test_security_attacks.py::TestPromptInjectionInstructions -v
uv run pytest tests/test_security_attacks.py::TestJailbreakAttacks -v

# 運行完整綜合測試（所有案例）
uv run pytest tests/test_security_attacks.py::TestAllSecurityCases -v
```

## Docker 環境測試

### 使用 Docker Compose

```bash
# 運行所有測試
docker compose --profile test run --rm test

# 運行特定測試
docker compose --profile test run --rm test sh -c "uv run pytest tests/test_input_guard.py -v"

# 運行安全測試
docker compose --profile test run --rm test sh -c "uv run pytest -m security -v"
```

## 測試覆蓋率

### 生成覆蓋率報告

```bash
# 生成 HTML 報告
uv run pytest --cov=chatbot_rag --cov-report=html

# 在瀏覽器中查看
open htmlcov/index.html
```

## CI/CD 集成

### GitHub Actions 範例

```yaml
- name: Run tests
  run: |
    uv sync --frozen --dev
    uv run pytest -v

- name: Run security tests
  run: |
    uv run pytest -m security -v

- name: Run comprehensive attack tests
  run: |
    uv run pytest tests/test_security_attacks.py::TestAllSecurityCases -v
```

## 常見測試命令速查

```bash
# 基礎測試
uv run pytest                                    # 運行所有測試
uv run pytest -v                                 # 詳細輸出
uv run pytest -v -s                              # 顯示 print 輸出
uv run pytest -x                                 # 遇到第一個失敗就停止

# 標記測試
uv run pytest -m security                        # 所有安全測試
uv run pytest -m "security and not slow"         # 安全測試（排除慢速）
uv run pytest -m injection                       # 注入測試
uv run pytest -m legitimate                      # 合法問題測試

# 特定測試
uv run pytest tests/test_input_guard.py         # 輸入驗證測試
uv run pytest tests/test_security_attacks.py    # 攻擊測試
uv run pytest -k "test_legitimate"              # 包含 legitimate 的測試

# 重新運行
uv run pytest --lf                               # 只運行上次失敗的
uv run pytest --ff                               # 先運行失敗的

# 覆蓋率
uv run pytest --cov=chatbot_rag                 # 顯示覆蓋率
uv run pytest --cov=chatbot_rag --cov-report=html  # 生成 HTML 報告

# 並行執行（需要 pytest-xdist）
uv add --dev pytest-xdist
uv run pytest -n auto                            # 自動使用所有 CPU 核心
```

## 測試數據

### attack_test_cases.json

完整的攻擊測試案例定義，包含：
- 13 個攻擊類別
- 60+ 個測試案例
- 涵蓋 Critical、High、Medium、Low 風險等級

測試案例會自動被 `conftest.py` 載入並提供給測試使用。

## 預期結果

### 理想測試結果

- ✅ **API 測試**: 100% 通過
- ✅ **輸入驗證測試**: 100% 通過
- ✅ **Critical/High 風險攻擊**: 100% 被阻止
- ✅ **合法問題**: 0% 誤判

### 測試失敗處理

如果測試失敗：

1. **查看詳細輸出**:
   ```bash
   uv run pytest tests/test_security_attacks.py::TestAllSecurityCases -v -s
   ```

2. **檢查失敗原因**: 查看測試輸出中的失敗詳情

3. **修復問題**:
   - 如果是攻擊未被阻止：更新 `input_guard_service.py` 中的檢測模式
   - 如果是合法問題被誤判：調整相關性檢查或檢測規則

4. **重新運行測試**:
   ```bash
   uv run pytest --lf -v
   ```

## 添加新測試

### 添加新測試案例到 attack_test_cases.json

```json
{
  "category": "你的類別",
  "cases": [
    {
      "id": "NEW-001",
      "description": "測試描述",
      "question": "測試問題",
      "expected_result": "應被阻止",
      "risk_level": "high"
    }
  ]
}
```

### 添加新測試類別到 test_security_attacks.py

```python
@pytest.mark.security
@pytest.mark.comprehensive
class TestNewCategory:
    """Test new category."""

    def test_all_new_category_cases(self, client, attack_test_cases):
        """Test all new category cases."""
        category_data = next(
            (c for c in attack_test_cases["test_cases"]
             if c["category"] == "你的類別"),
            None
        )

        if not category_data:
            pytest.skip("No test cases found")

        failures = []
        for case in category_data["cases"]:
            response = client.post(
                "/api/v1/rag/ask",
                json={"question": case["question"]}
            )

            assert response.status_code == 200

            result = response.json()
            passed, reason = evaluate_result(
                result, case["expected_result"], case["risk_level"]
            )

            if not passed:
                failures.append(f"{case['id']}: {reason}")

        assert not failures, f"Failed cases:\n" + "\n".join(failures)
```

## 疑難排解

### 測試無法連接到 API

確保 API 服務正在運行：

```bash
# Docker 環境
docker compose up -d app-dev

# 本地環境
uv run chatbot-dev
```

### attack_test_cases.json 未找到

確保文件存在於專案根目錄：

```bash
ls -la attack_test_cases.json
```

### LLM 服務不可用

某些測試需要 LLM 服務（相關性檢查）。如果 LLM 不可用：

1. 測試可能會失敗或變慢
2. 可以臨時禁用相關性檢查：
   ```bash
   ENABLE_RELEVANCE_CHECK=False uv run pytest
   ```

## 更多信息

- 📖 [輸入驗證指南](../docs/INPUT_GUARD.md)
- 🛡️ [攻擊測試指南](../docs/ATTACK_TESTING.md)
- 📚 [快速參考](../ATTACK_TEST_QUICK_START.md)
