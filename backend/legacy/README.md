# backend/legacy — 历史一次性产物归档

这里放的是项目早期（2026-08 前后）在 `backend/` 根目录随手留下的**一次性调试脚本、
临时夹具与审计报告**。它们已经没有任何代码引用（`app/`、`tests/`、`scripts/`、
compose 与前端都不依赖），保留只是为了"哪天想回看当时的排查方式时还在"。

**归档而不是删除**，是因为其中几份记录了当初怎么定位问题的（例如并发重复上传、
zip 炸弹、内存占用），比结论本身更有参考价值。

## 为什么必须挪走

1. **污染 pytest 收集**：`test_all.py` / `test_memory.py` / `test_retrieval.py` 这类
   名字符合 `test_*.py` 约定，从 `backend/` 根目录直接跑 `pytest` 时会被当成测试
   收集并执行 —— 实际它们是需要手工触发的调试脚本。真正的测试套件在 `../tests/`。
2. **和正式测试抢名字**：`test_duplicate.py` 与 `../tests/test_duplicate.py` 同名，
   容易让人改错文件。

## 内容速查

| 文件 | 当时用来干什么 |
|---|---|
| `audit_tests.py` / `comprehensive_audit.py` / `final_audit_report.md` | 早期一轮自审 |
| `test_all_formats.py` / `generate_test_files.py` / `test_files/` | 造多格式夹具（它自己生成 `test_files/`） |
| `test_duplicate.py` / `test_concurrent_dup.py` / `test_zip_bomb.py` | 判重、并发重复、解压炸弹 |
| `test_memory.py` / `test_memory_debug.py` / `test_large.txt` | 大文件内存占用 |
| `test_retrieval.py` / `test_retrieval_2.py` | 早期检索手工验证 |
| `upload.py` / `*.pdf`（除 `../test.pdf`） | 手工上传脚本与随手样本 |
| `clean_orphans.py` | 清理孤儿向量 |
| `debug/` | 分块与 PDF 解析的一次性调试脚本 |

## 注意

- `../test.pdf` **没有**被归档：它是 `../tests/test_upload.py` 的上传冒烟夹具，
  该用例按 CWD 相对路径找它。
- 需要当前仍在维护的运维/诊断脚本，请看 `../scripts/`（`diag_chunks.py`、
  `diag_image_payload.py`、`mint_token.py`、`reingest_shizhan.py` 等）。
