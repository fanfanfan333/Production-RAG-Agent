# 增量 PRD：用户请求级企业 RAG 数据隔离（五维权限模型）

> 类型：已有系统增量需求（简单增量 PRD，不含竞品/市场分析）
> 语言：简体中文
> 范围：只描述**规则与语义**（产品侧），不含实现代码与技术方案
> 基线：D:\RAG\Production-RAG-Agent 现有系统

---

## 0. 基线继承说明（先说清楚什么不动）

本 PRD 是**纯增量**。以下内容**继承现有实现，不在本轮变更范围内，不做任何删除或替换**：

| 已有能力 | 继承来源 | 本轮关系 |
|---|---|---|
| 三层知识库 private / department / company | `knowledge_tier_service.py`、`tenancy.ACCESS_PRIVATE/DEPARTMENT/TENANT` | **保留原样**，新增维度叠加在其上 |
| RBAC 角色矩阵（employee / dept_manager / kb_admin / company_admin / admin + 历史角色映射） | `permissions.py` | **保留原样**，本轮只定义"角色与密级/项目的关系" |
| 租户隔离与可见范围组装 `scope_for` / `DocumentScope` / `TENANT_WIDE_READER_ROLES` | `tenancy.py` | **扩展**（增加 clearance / project_ids / principals），不重写 |
| Vector / BM25 / RRF / Rerank 主链路 | `retrieval_service.py` / `hybrid_search.py` / `reranker.py` | **保留**，在两路检索前置同一份 Scope 过滤 |
| 多模态图片理解管线、文档解析器 | `image_understanding/` / `parsers/` | **保留**，为其产物定义权限继承规则 |
| 共享申请 ShareRequest（pending/approved/rejected/cancelled） | `share_service.py`、`db/share_models.py` | **保留**，本轮定义其在隔离链路中的中间态语义 |
| 审计 `audit_service.py` | — | **扩展**审计事件类型 |

**已确认缺口（本轮补齐）**：`security_level` / `clearance` / 密级 —— 全库不存在；`project_id` / 项目维度 —— 不存在独立的"项目"授权维度（现有 `project_id` 仅出现在无关语义中）。

> 与现有实现的衔接约束（供架构师参考，非实现指令）：现有向量层过滤 `_visibility_conditions` 采用的是 **deny-list + fail-open** 形态（字段缺失的老向量不被排除，交由 PG 的 `document_scope_clause` 兜底）。新增的密级与项目维度一旦加入，**"payload 副本过期"的窗口客观存在**，因此本 PRD 把"检索后 ACL 复核以 PG 为权威源"列为 P0 而非 P1 —— 这不是重复设计，而是新增维度后的必需补偿。

---

## 1. 产品目标与验收标准

### 1.1 目标

1. **请求级绑定**：任何一次查询，从进入检索器的第一刻起就携带一份不可变的 `UserScope`；链路中任何环节都不得重新查库放宽它。**禁止"先检索全库、再靠 Prompt 做权限隔离"**。
2. **五维可表达**：租户 / 部门 / 角色 / 密级 / 项目 五个维度都能在对象 metadata 上被表达，且关系可判定、冲突可消解。
3. **权限一致继承**：文档 → 文本 Chunk / 表格 / 代码块 / 图片对象，权限继承关系一致且只收紧不放宽；**OCR 把图片变成可检索文本时不得产生新的泄密面**。
4. **可审计**：任何被权限剔除的对象都留痕，可回答"谁、在哪个环节、因为什么、被剔除了什么"。

### 1.2 验收标准（可判定）

| 编号 | 验收条件 | 判定方式 |
|---|---|---|
| **A1 全链路无泄漏** | 低密级用户提问时，在「Vector 候选 / BM25 候选 / RRF 后 / Rerank 后 / Context 组装后 / LLM 输入 / 最终回答 / 引用列表」**八个采样点**，均不出现 `effective_security_level > user.clearance` 的对象（含图片对象及其 OCR 派生文本） | 对同一 query 打点采样，逐点断言 |
| **A2 双路同源** | Vector 路与 BM25 路使用的 Scope 过滤表达式**序列化结果逐字节相同** | 序列化比对断言 |
| **A3 收紧立即生效** | 管理员把文档密级从 1 提到 3（或加入 `acl_deny`）后，**在级联任务完成前**发起查询，结果中不得出现该文档的任何派生对象 | 人工制造级联延迟后查询 |
| **A4 越权留痕** | 第 7 / 11 / 12 环剔除的每个对象，产生一条审计记录（user_id / object_id / 剔除环节 / 原因），可查询 | 审计查询断言 |
| **A5 图文一致** | 某图片单独提级到 3，`clearance=1` 的用户既**看不到原图回显**，也**检索不到该图的 OCR 文本 / 表格 / 代码块** | 图文分别断言 |
| **A6 项目横向** | 跨部门项目成员可检索到 `visibility_mode=project` 且 `project_ids` 命中的文档；**同部门但非项目成员不可** | 两个账号对照 |
| **A7 中间态** | 共享申请 `pending` 期间，申请人以外（含管理员）不可见；`approved` 后立即可见；`rejected`/`cancelled` 后不可见 | 四态逐一断言 |
| **A8 角色不越密** | `kb_admin` 的 clearance 被下调为 1 后，看不到 secret 文档；`admin` 同样看不到高于自身 clearance 的文档 | 改配置后查询 |
| **A9 个人库红线保持** | 上述全部改动上线后，他人 `private` 文档仍仅本人可见（admin 例外**仅限**自建测试公司，与现有 `owns_tenant_ids` 语义一致） | 回归断言 |
| **A10 拒绝而非编造** | 全部候选被剔除时，回答必须是明确的"无可用资料 / 权限不足"，**不得生成任何内容** | 构造全剔除场景 |

---

## 2. 五维权限模型（本 PRD 核心）

### 2.1 一句话总纲

> **OR 用于"范围来源"，AND 用于"硬性闸门"，deny 覆盖 allow。**

### 2.2 五者的角色分工

| 维度 | 性质 | 在判定式中的位置 | 说明 |
|---|---|---|---|
| **租户 tenant** | **硬性闸门（AND）** | 最外层，先过滤 | 一切判定的容器；跨租户是唯一不可跨越的边界（平台管理员按现有收敛口径除外） |
| **部门 department** | **范围来源（OR 之一）** | 租户内 | 命中"同部门"即获得该文档的可见性候选资格 |
| **项目 project** | **范围来源（OR 之一）** | 租户内，**横向于部门** | 命中"项目成员"同样获得候选资格；与部门**互不隶属** |
| **密级 security_level** | **硬性闸门（AND）** | 范围来源之上叠加 | `user.clearance >= obj.effective_security_level` 才放行 |
| **角色 role** | **只管动作能力，不管数据可见性** | 不进入数据范围判定 | 与 `permissions.py` 现有注释精神一致："能力矩阵决定能不能做这个动作，`tenancy.scope_for` 决定这个动作能看到哪些数据" |

### 2.3 判定式（产品语义，非实现）

```
可见(obj, user) =
      tenant_gate        : obj.tenant_id ∈ user.tenant_ids                    [AND, 硬]
  AND security_gate      : user.clearance >= obj.effective_security_level
                           或 obj 对用户存在未过期的 need-to-know 例外          [AND, 硬]
  AND source_gate        : (                                                  [OR, 范围来源]
          owner_hit      : obj.owner_id == user.id
       OR dept_hit       : obj.access_level == "department"
                           AND (obj.department_id == user.department_id OR user.tenant_wide)
       OR tenant_hit     : obj.access_level == "tenant"
       OR project_hit    : obj.visibility_mode == "project"
                           AND obj.project_ids ∩ user.project_ids ≠ ∅
       OR acl_hit        : user.principals ∩ obj.acl_allow ≠ ∅
                           AND (obj.acl_expires_at IS NULL OR 未过期)
  )
  AND deny_gate          : user.principals ∩ obj.acl_deny == ∅                [AND, 最高优先级]
```

**冲突消解规则（按优先级从高到低）**：

1. **`acl_deny` 绝对优先**：命中即拒绝，其余维度全部失效（包括管理员、包括所有者本人）。
2. **硬闸门先于范围来源**：租户不匹配 / 密级不足时，**无论来源多充分都拒绝**。不存在"因为是项目成员所以可以看高密级"。
3. **范围来源之间取并集**：命中任一来源即可获得候选资格（这是"或"）。
4. **密级取严**：派生对象的有效密级 = `max(自身密级, 父对象密级)`。
5. **字段缺失 fail-closed**：密级缺失按待确认 Q2 的默认值处理（默认 1）；范围来源字段缺失即视为不命中任何来源 → 不可见。

### 2.4 密级建模

**两侧建模**：

- **文档侧 `security_level`**（对象属性）：该对象本身的密级档位。
- **用户侧 `clearance`**（许可上限）：该用户可接触的最高密级。

**比较规则**：`user.clearance >= obj.effective_security_level` → 放行；否则拒绝。

**推荐 4 档（见待确认 Q1）**：

| 值 | 名称 | 典型内容 |
|---|---|---|
| 0 | 公开 public | 公司公告、对外宣传材料 |
| 1 | 内部 internal | 一般制度、流程文档（**存量默认值**） |
| 2 | 保密 confidential | 部门经营数据、客户名单 |
| 3 | 机密 secret | 薪酬、合同、未披露财务 |

**角色 → 默认 clearance（可配置，仅作初值，不是豁免）**：

| 角色 | 默认 clearance | 说明 |
|---|---|---|
| employee | 1 | — |
| dept_manager | 2 | — |
| kb_admin / company_admin | 3 | — |
| admin（平台管理员） | 3 | **仍是有上限的"高"，不是"无限"** |

**need-to-know（需知所需）例外**：**需要，但必须受限**。规则：

1. 例外只在**对象级 `acl_allow` 显式授予**时生效（授予主体可是 user / dept / role / project / group）。
2. 例外**必须带有效期** `acl_expires_at`；为空表示长期，但每次命中都写审计。
3. 例外**只能由具备审批动作权限的角色授予**（沿用 `share.review.department` / `share.review.company` 的审批链），**不允许自我授予**。
4. **角色不产生隐式 need-to-know**：没有任何角色"自动"获得超密级访问权限。

### 2.5 项目维度

- **项目是独立于部门之外的横向维度，不是部门的子集。**
- 项目归属**唯一租户**（`project.tenant_id`），不跨租户；成员可**跨部门**。
- 跨部门项目组：成员身份只看 `user.project_ids`，**部门维度不参与判定**（既不需要同部门，也不因同部门而自动获得）。
- 表达方式（推荐项，见待确认 Q3）：**不改 `access_level` 三值**，新增 `project_ids` 字段 + `visibility_mode` 枚举：
  - `visibility_mode = "tier"`（默认）：按现有三层语义判定，项目字段不参与 —— **存量文档行为零变化**。
  - `visibility_mode = "project"`：仅 `project_ids` 命中者可见（仍需通过租户 + 密级闸门）。

选择该表达方式的理由：① 不动现有三层语义与前端标签体系（符合"额外多的部分不要删除"）；② 项目与层级正交，一份文档可同时"属于部门库 + 属于某项目"；③ 存量缺失 `project_ids` 时不匹配任何项目，退回原三层判定，**不会 fail-open**。

### 2.6 三类管理员的豁免与不豁免（必须与现有代码口径一致）

| 维度 | 平台管理员 admin | 企业管理员 company_admin | 知识库管理员 kb_admin |
|---|---|---|---|
| 租户 | ✅ 跨租户，**但仅限所属租户 ∪ 自建测试公司**（继承现有 `owns_tenant_ids` 收敛口径，非全平台） | ❌ 仅本公司 | ❌ 仅本公司 |
| 部门 | ✅ `tenant_wide` 全通 | ✅ `tenant_wide` 全通 | ✅ `tenant_wide` 全通 |
| **他人个人库 private** | ❌ **不能看**（唯一例外：其**自建测试公司**内的成员 private 文档，沿用现有 `owns_tenant_ids` 窄口径） | ❌ **不能看** | ❌ **不能看** |
| 密级 | ❌ **不豁免**。默认 clearance=3（有上限），超出自身 clearance 的文档同样不可见。**可以授予**他人密级例外，但**不能自动拥有** | ❌ 不豁免。默认 3，可在**本公司范围内**审批密级授予 | ❌ 不豁免。默认 3，可在**本公司范围内**审批密级授予 |
| 项目 | ❌ **不自动成为**任何项目成员；不豁免 | ❌ 不自动成为成员；可**审批**本公司项目成员申请 | ❌ 不自动成为成员；可**审批**本公司项目成员申请 |
| `acl_deny` | ❌ **不豁免**（命中即拒绝，高于一切） | ❌ 不豁免 | ❌ 不豁免 |

> 设计意图：管理员账号被攻破是最现实的泄密路径。让"管理员 = 无限可见"等于把整个密级体系的保险丝短接。本模型下管理员拥有**授权能力**，但不拥有**自动可见性**。

---

## 3. 权限继承规则

### 3.1 对象类型

`doc`（文档）→ 派生出 `text_chunk`（文本块）/ `table`（表格）/ `code`（代码块）/ `image`（图片对象）。

### 3.2 继承语义

**默认：完全继承。派生对象不携带独立 ACL，其权限 = 父文档权限。**
（落库时冗余写入 payload 以保证向量层可过滤，但**权威源始终是 PG 的文档记录**。）

**唯一允许的偏离方向：收紧，不允许放宽。**

```
effective(obj) = parent(doc) ∩ obj_self_tightening
```

判定细则：

| 维度 | 派生对象可做什么 | 不可做什么 |
|---|---|---|
| 密级 | 可**上调**（`security_level` 高于父文档） | ❌ 不得下调；若出现下调，以父文档为准（取严） |
| 租户 / 部门 / 所有者 | ❌ 不得偏离父文档 | 一律继承 |
| 项目 | 可**加闸**（要求必须同时是项目成员） | ❌ 不得把父文档的项目限制放宽 |
| ACL | 可**加** `acl_deny` | ❌ 不得通过 `acl_allow` 获得父文档之外的可见性 |

### 3.3 图片对象

**默认从文档继承。** 以下四种情形**需要图片级收紧**：

1. 图片含敏感实体（身份证 / 银行卡 / 工牌 / 签名 / 印章）—— 可由现有 `image_understanding/classifier.py` 的识别结果触发；
2. 图片本身来自更高密级的来源（如扫描件内含机密表格）；
3. 文档共享申请被批准，但审批人要求隐去其中某张图；
4. 图片被多份文档引用，其中存在更高密级的引用方。

**图片级收紧的三种形态**（P0 只做前两种，见待确认 Q4）：

- **a. 提级**：`security_level` 上调 → 低 clearance 用户看不到该图片对象；
- **b. 剔除**：该图片不进入任何检索与上下文，文档其余部分照常（这是"脱敏"而非"收紧"）；
- **c. 独立 ACL 编辑器**（P2，本期不做）。

### 3.4 表格 / 代码块

- **表格**作为 chunk（`content_type="table"`）继承文档权限。表格是数据密集区，**允许表格级提级**（如含薪资列的表）。
- **代码块**若为**图片 OCR 派生**（`is_image_derived == true`），权限取严：
  ```
  effective(code_ocr) = max(文档权限, 源图片权限)
  ```
- **红线**：**OCR 产出的文本 / 表格 / 代码，不得比源图片更宽松。** 否则会出现"图片看不了，但能搜到图片里的字"的破口 —— 这是本 PRD 中最容易被忽略、后果最严重的继承漏洞。

### 3.5 级联策略

**总原则：收紧同步，放宽异步。**

| 变更方向 | 时效要求 | 机制 |
|---|---|---|
| **收紧**（提密级、加 deny、移除项目成员、移出部门） | **立即生效，不得有窗口** | 检索后 ACL 复核以 PG 权威源为准，**不信任 payload 副本** |
| **放宽**（降密级、加项目成员、批准共享） | 允许**秒级~分钟级最终一致** | 异步级联任务更新所有派生对象 payload，带进度与失败重试 |

一致性要求：

1. **双写**：落库时把有效权限写入每个派生对象的 payload（保证向量层可前置过滤）；
2. **级联**：文档权限变更触发异步任务刷新派生对象；`acl_sync_state` 记录 `synced / pending / stale`；
3. **兜底**：`acl_sync_state != synced` 时，第 11 环复核**必须**以 PG 权威源判定，并记入审计；
4. **反向指针**：`derived_object_ids` 让文档变更能定位全部派生对象，避免全表扫描。

---

## 4. Metadata Schema（业务层面定义）

字段命名沿用仓库现有 snake_case 风格；Qdrant payload 保持**扁平结构**（与现有 `vector_service.py` 注释一致：嵌套路径不利于建索引）。

### 4.1 对象侧字段表

**A. 主体与租户（硬边界）**

| 字段 | 类型 | 取值范围 | 默认 | 说明 |
|---|---|---|---|---|
| `tenant_id` | string(keyword) | 租户 ID，如 `"c8111de986583"` | 必填 | 硬边界，AND |
| `owner_id` | uuid string | 用户 ID | 必填 | 所有者（文档上传者） |
| `user_id` | uuid string | 用户 ID | 必填 | **现有 Qdrant payload 的所有者字段名**（= `owner_id` 的向量层副本）。保留原名，避免破坏既有过滤条件 |
| `department_id` | string \| null | 部门 ID，如 `"d_marketing"` | null | 归属部门；部门库文档必填 |

**B. 层级与可见范围**

| 字段 | 类型 | 取值范围 | 默认 | 说明 |
|---|---|---|---|---|
| `access_level` | enum | `private` / `department` / `tenant` | `private` | **现有三层，保留不变** |
| `visibility_mode` | enum | `tier` / `project` | `tier` | 新增。`tier`=按三层判定（存量行为不变）；`project`=仅项目成员 |
| `project_ids` | list[string] | 项目 ID 集合 | `[]` | 新增。横向维度，与部门正交 |
| `visible_scope` | enum | `self` / `department` / `tenant` / `project` / `acl` | 派生 | **派生只读字段**，由前三者推导，供快速过滤与前端标签，**不可单独写入** |

**C. 密级**

| 字段 | 类型 | 取值范围 | 默认 | 说明 |
|---|---|---|---|---|
| `security_level` | int | `0..3` | `1` | 对象自身密级（见待确认 Q2） |
| `parent_security_level` | int \| null | `0..3` | null | 父对象密级（派生对象冗余，用于级联与审计） |
| `effective_security_level` | int | `0..3` | 派生 | `= max(security_level, parent_security_level)`。**检索层实际使用此字段** |

**D. ACL 主体**

| 字段 | 类型 | 取值范围 | 默认 | 说明 |
|---|---|---|---|---|
| `acl_allow` | list[string] | `"user:<uuid>"` / `"dept:<id>"` / `"role:<role>"` / `"project:<id>"` / `"group:<id>"` | `[]` | 显式允许（need-to-know 例外的载体） |
| `acl_deny` | list[string] | 同上格式 | `[]` | 显式拒绝，**优先级高于一切，含管理员与所有者本人** |
| `acl_expires_at` | datetime \| null | ISO8601 | null | 临时授权到期时间（共享 / 项目临时成员 / need-to-know） |

**E. 对象类型与继承来源**

| 字段 | 类型 | 取值范围 | 默认 | 说明 |
|---|---|---|---|---|
| `object_type` | enum | `doc` / `text_chunk` / `table` / `code` / `image` | 必填 | 对象类型 |
| `content_type` | string | 现有取值（text/table/code/image…） | — | **现有字段，保留** |
| `parent_object_id` | string \| null | 对象 ID | null | 父对象（chunk/table/code/image 的父 = `document_id`） |
| `inherited_from` | string \| null | 对象 ID | null | 权限继承来源对象 ID |
| `inherited_at` | datetime \| null | ISO8601 | null | 继承快照时间 |
| `acl_sync_state` | enum | `synced` / `pending` / `stale` | `synced` | 级联同步状态；非 `synced` 时复核须走 PG 权威源 |
| `derived_object_ids` | list[string] | 对象 ID | `[]` | 反向指针（仅 doc 维护），用于级联定位 |

**F. 共享中间态**

| 字段 | 类型 | 取值范围 | 默认 | 说明 |
|---|---|---|---|---|
| `share_status` | enum | `none` / `pending` / `approved` / `rejected` / `cancelled` | `none` | 与 `ShareRequest.status` 对齐 |
| `share_grant_scope` | enum \| null | `department` / `tenant` / `project` | null | `approved` 后的生效范围 |

### 4.2 请求级载体：UserScope 运行态字段表

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | uuid string | 主体 |
| `tenant_ids` | frozenset[string] | 租户集合（继承现有 `DocumentScope.tenant_ids` 语义） |
| `home_tenant_id` | string | 所属租户 |
| `department_id` | string \| null | 部门 |
| `tenant_wide` | bool | 本租户内部门库全通（继承现有语义） |
| `role` | string | 角色 —— **只用于动作能力，不参与数据范围判定** |
| `clearance` | int `0..3` | 许可密级上限 |
| `project_ids` | frozenset[string] | 项目成员集合 |
| `principals` | list[string] | 主体标识集合，如 `["user:u1","dept:d_marketing","role:kb_admin","project:p_alpha"]`，用于 `acl_allow` / `acl_deny` 匹配 |
| `owns_tenant_ids` | frozenset[string] | 自建测试公司集合（继承现有语义，决定 private 例外范围） |
| `scope_fingerprint` | string | 五维指纹（继承现有 `tenant_scope_fingerprint`，扩展为五维），用于缓存细分与日志 |
| `issued_at` | datetime | 签发时间 |

> **不可变性要求**：`UserScope` 在请求入口一次性生成，链路内**只读**。任何环节不得重新查库"补权限"；需要变更时必须重新发起请求。

### 4.3 完整示例：文档对象

```json
{
  "object_type": "doc",
  "object_id": "8f2c4a10-5d3e-4b7a-9c11-2e6f8a0b4d31",
  "document_id": "8f2c4a10-5d3e-4b7a-9c11-2e6f8a0b4d31",
  "tenant_id": "c8111de986583",
  "owner_id": "u_1001",
  "user_id": "u_1001",
  "department_id": "d_marketing",
  "access_level": "department",
  "visibility_mode": "tier",
  "project_ids": [],
  "visible_scope": "department",
  "security_level": 2,
  "parent_security_level": null,
  "effective_security_level": 2,
  "acl_allow": ["dept:d_marketing", "user:u_1002"],
  "acl_deny": ["user:u_1009"],
  "acl_expires_at": null,
  "parent_object_id": null,
  "inherited_from": null,
  "inherited_at": null,
  "acl_sync_state": "synced",
  "derived_object_ids": [
    "chk_8f2c_0017",
    "tbl_8f2c_p07_02",
    "img_8f2c_p07_02",
    "cod_8f2c_p12_01"
  ],
  "share_status": "none",
  "share_grant_scope": null,
  "filename": "2025_Q1薪酬方案.pdf",
  "content_type": "pdf"
}
```

### 4.4 完整示例：图片对象（被单独收紧）

```json
{
  "object_type": "image",
  "object_id": "img_8f2c_p07_02",
  "document_id": "8f2c4a10-5d3e-4b7a-9c11-2e6f8a0b4d31",
  "tenant_id": "c8111de986583",
  "owner_id": "u_1001",
  "user_id": "u_1001",
  "department_id": "d_marketing",
  "access_level": "department",
  "visibility_mode": "tier",
  "project_ids": [],
  "visible_scope": "department",
  "security_level": 3,
  "parent_security_level": 2,
  "effective_security_level": 3,
  "acl_allow": ["user:u_1002"],
  "acl_deny": [],
  "acl_expires_at": "2025-12-31T23:59:59+08:00",
  "parent_object_id": "8f2c4a10-5d3e-4b7a-9c11-2e6f8a0b4d31",
  "inherited_from": "8f2c4a10-5d3e-4b7a-9c11-2e6f8a0b4d31",
  "inherited_at": "2025-03-18T10:22:41+08:00",
  "acl_sync_state": "synced",
  "derived_object_ids": ["tbl_8f2c_p07_02"],
  "share_status": "none",
  "share_grant_scope": null,
  "image_id": "img_8f2c_p07_02",
  "image_path": "documents/8f2c4a10/images/p07_02.png",
  "page_number": 7,
  "position": 2,
  "bbox": [72.5, 188.0, 523.0, 402.5],
  "content_type": "image",
  "image_type": "table",
  "analyze_engine": "docling",
  "analyze_confidence": 0.87,
  "manual_review": false
}
```

> 说明：该图片被从文档的 2 级（保密）**单独提级到 3 级（机密）**，并对 `u_1002` 授予了带有效期的 need-to-know 例外。其 OCR 派生的表格 `tbl_8f2c_p07_02` 的 `effective_security_level` **必须为 3**（取严，见 3.4 红线）。

---

## 5. 十二环节逐环节产品要求

### 环节 1 · 用户身份认证 Authentication

1. 认证成功后必须解析出**五项主体属性**：`tenant_ids` / `department_id` / `role` / `clearance` / `project_ids`；任一项解析失败时按**最小权限**处理（不得按"宽松"放行）。
2. 未认证请求不得进入后续任何环节（fail-closed，沿用现有行为）。

### 环节 2 · 授权 Authorization

1. **角色只判定"动作能力"**（沿用 `permissions.py` 现有矩阵），**不产出数据可见范围**。
2. 密级不足 / 非项目成员 **不改变**用户的动作能力判定结果：用户仍"可以发起检索"，只是结果为空。**返回空结果，而不是 403 报错** —— 避免通过错误码探测文档存在性。

### 环节 3 · Scope 生成 UserScope

1. 在请求入口**一次性生成不可变** `UserScope`（字段见 4.2），链路内只读；**禁止任何环节重新查库放宽**。
2. `UserScope` 缺失或不完整时，结果为空集，不得降级为"不过滤"。
3. `scope_fingerprint` 必须覆盖全部五个维度；缓存 key 必须包含该指纹（**不同 Scope 的请求不得共用缓存结果**）。

### 环节 4 · Query

1. **Query 进入检索器时必须已绑定 `UserScope`**；不存在"无 Scope 的检索调用"。
2. 用户在 Query 上声明的元数据过滤（`MetadataFilter`，如年份 / 文档类型 / 指定文档）**只能缩小、不能扩大** Scope —— 最终条件 = `UserScope ∩ 用户过滤`。
3. **禁止**依赖 Prompt / System Message 做权限隔离（红线）。

### 环节 5 · 文档 ACL

1. **PG 中的文档记录是权限权威源**；Qdrant payload 中的权限字段是**副本**，不是权威。
2. 每个文档必须完成五维标注（租户 / 部门 / 层级 / 密级 / 项目）；未标注密级的按待确认 Q2 默认值处理。
3. 文档权限变更必须产生审计事件与级联任务（见 3.5）。

### 环节 6 · 图片 ACL

1. 图片对象**默认继承**文档权限；支持图片级**收紧**（提级 / 剔除），**不支持放宽**。
2. 图片被收紧后，其 OCR 派生的文本 / 表格 / 代码块**同步收紧**（取严红线，A5）。
3. 图片对象被剔除时，该图片**不得**出现在任何上下文与引用回显中，且回答中不得描述其内容。

### 环节 7 · 检索前权限过滤（**本环节是隔离主闸门**）

1. **Vector 路与 BM25 路必须使用同一份由 `UserScope` 生成的过滤条件**，实现上同源，禁止两路各自拼装。
2. 过滤发生在**召回之前**（ANN / 倒排之内），不是"先召回后剔除"。
3. 过滤条件构造失败或异常时 → **fail-closed 返回空**，不得降级为"先召回后过滤"。
4. 产品意图：保证候选池里**从一开始就不存在**越权对象，而不是事后打捞。

### 环节 8 · Vector / BM25 双路检索

1. 两路各自独立召回，但**共享同一份 Scope 表达式**；任一路不得绕过 Scope 走"全库召回"。
2. 两路的候选深度、过滤字段语义必须一致；不一致视为缺陷（A2）。

### 环节 9 · RRF 融合

1. 融合的输入**只能是已通过第 7 环**的候选；融合过程**不得引入**任何未经 Scope 过滤的新候选。
2. RRF 的权重与打分**不得**把权限字段作为特征（避免通过排序泄露"存在高密级内容"）。

### 环节 10 · Rerank

1. CrossEncoder 只接收已过 Scope 的候选；输出 Top20 仍须送第 11 环复核。
2. Rerank 只评估语义相关性；**不得**因权限字段调整排序。

### 环节 11 · 检索后 ACL 校验（**权威复核**）

1. 逐个候选以 **PG 权威源**复核（防 payload 副本过期 / 级联未完成 / 权限刚被收紧）。
2. 复核不通过 → **剔除 + 审计留痕**（A3、A4）。
3. 剔除后剩余不足 → 走第 12 环的降级策略，**不得用被剔除内容兜底**，也不得"降低阈值再召回一批"。

### 环节 12 · Context 组装 → LLM 输入前最终校验 → 引用溯源

1. **Context Builder** 只组装通过第 11 环的块；`TextBlock` 与 `ImageBlock` **分别**校验（两者权限可能不同）。
2. **Final Check**：组装完成后、送入 LLM 前，逐块再与 `UserScope` 比对一次；任何一块不通过 → 剔除该块（并审计）。
3. **图片回显**：只有"原文图片（Original Image）"可作为引用回显，且必须通过图片级 ACL；**禁止用图片描述文本替代图片回显**来绕过图片 ACL。
4. **引用溯源**：每条引用必须携带 `document_id` + 定位信息（page / line / bbox / position）+ **该块的权限快照**；用户**点击引用打开时须再次校验**（打开时权限可能已变更）。
5. **降级策略（本环节核心）**：

| 情形 | 策略 | 禁止事项 |
|---|---|---|
| 全部候选被剔除 | **拒绝回答**：明确输出"没有可用资料 / 权限不足"，不编造内容 | ❌ 不得给"大致意思"、不得推测、不得复述问题 |
| 部分被剔除 | **部分回答**：用剩余证据作答，并在结尾标注"部分内容因权限限制未包含" | ❌ 不得泄露被剔除文档的名称 / 数量 / 存在性（见待确认 Q6） |
| 图片被剔除但文本可用 | **文本回答 + 不回显该图片** | ❌ 不得在回答中描述被剔除图片的内容 |
| 权限快照缺失 | **按不通过处理**（fail-closed） | ❌ 不得因"不知道权限"而放行 |

---

## 6. 用户故事

1. **跨租户**：作为 A 公司的普通员工，我要我的检索结果**永远不会包含 B 公司的任何文档**，以便公司之间的数据边界是硬的，而不是靠 Prompt 约束。

2. **跨部门**：作为市场部成员，我要**看不到研发部的部门库文档**，但**能看到公司库文档**，以便部门信息不横向流失。

3. **密级不足**：作为 clearance=1 的普通员工，我在问"今年的薪酬方案"时，要**既搜不到薪酬文档的文本块，也看不到其中的表格图片**，以便密级闸门对文本和图片同等生效（而不是只管文本）。

4. **项目组内（跨部门）**：作为"阿尔法项目"的成员（我属市场部、同事属研发部），我要能检索到该项目的全部项目文档，**无论我与文档作者是否同部门**，以便跨部门协作不受部门墙阻挡。

5. **项目外（反向）**：作为与项目文档作者**同部门但非项目成员**的员工，我要**看不到**该 `visibility_mode=project` 的项目文档，以便项目信息不因同部门而自动扩散。

6. **图片单独收紧**：作为文档作者，我要能**把文档中的某一张含身份证的截图单独提级或剔除**，而文档其余部分保持原可见范围，以便不必为了一张图把整份文档设为机密。

7. **OCR 不能成为后门**：作为安全负责人，我要**图片被收紧时，其 OCR 出来的文字/表格/代码同步收紧**，以便不会出现"图片看不了、但能搜到图片里的字"的破口。

8. **共享申请中间态**：作为提交共享申请的员工，我要在申请 `pending` 期间**该文档对他人仍不可见**（包括管理员），`approved` 后立即可见，`rejected`/`cancelled` 后恢复不可见，以便审批过程不产生权限真空。

9. **权限收紧立即生效**：作为知识库管理员，我把一份文档从"内部"提到"机密"后，要**正在进行的对话也立即看不到它**（不得等缓存/级联任务跑完），以便泄密窗口为零。

10. **管理员不越密**：作为企业管理员，我要我的密级是**可配置且有上限的**，以便我的账号被攻破时不会一次性泄出全公司机密。

---

## 7. 需求池（P0 / P1 / P2）

### P0（必须）

| 编号 | 需求 | 验收 |
|---|---|---|
| **P0-1** | **五维 UserScope**：认证侧产出 tenant / department / role / clearance / project 五项，请求入口一次性生成、链路内不可变 | 环节 1–3；A2 |
| **P0-2** | **密级建模与比较规则**：`security_level` vs `clearance`，deny-overrides，4 档，角色给默认值不豁免 | A1、A8 |
| **P0-3** | **项目维度（横向）**：`project_ids` + `visibility_mode`，跨部门成员判定，不动现有 `access_level` 三值 | A6 |
| **P0-4** | **Metadata Schema 落地**：PG 列 + Qdrant payload 双写，字段以第 4 节字段表为准，扁平结构 | 字段表逐项核对 |
| **P0-5** | **检索前双路 Scope 下推**：Vector 与 BM25 同源表达式，召回前过滤，异常 fail-closed | A2、A1 |
| **P0-6** | **检索后 ACL 权威复核**：以 PG 为权威源，剔除 + 审计 | A3、A4 |
| **P0-7** | **权限继承与级联**：派生对象只收紧不放宽；收紧同步、放宽异步；`acl_sync_state` | A3、A1 |
| **P0-8** | **图片级收紧 + OCR 派生取严**：提级 / 剔除；OCR 产物 `effective_security_level = max(文档, 源图)` | A5 |
| **P0-9** | **LLM 输入前最终校验 + 引用溯源**：Final Check 逐块校验；引用带权限快照；点击引用再校验；降级策略按第 12.5 表执行 | A10 |
| **P0-10** | **越权审计留痕**：第 7 / 11 / 12 环剔除均产生审计记录（谁 / 对象 / 环节 / 原因） | A4 |
| **P0-11** | **验收测试套件**：覆盖 A1–A10 全部断言，含跨租户、跨部门、密级不足、项目内外、图文一致、四态共享、A3 级联延迟场景 | 测试全绿 |

### P1（应该）

| 编号 | 需求 |
|---|---|
| **P1-1** | **need-to-know 例外**：对象级 `acl_allow` 显式授予 + 有效期 + 审批链，禁止自我授予 |
| **P1-2** | **项目临时成员**：项目成员可带 `expires_at`，到期自动失效 |
| **P1-3** | **存量密级回填工具**：未标注密级文档清单 + 批量标注 + "严格模式"开关（未标注按最高密级处理） |
| **P1-4** | **图片敏感自动提级**：接 `image_understanding/classifier.py` 的识别结果，对含证件/印章类图片自动提级（可人工复核） |
| **P1-5** | **Scope 可观测**：日志打印 `scope_fingerprint` 与被剔除对象计数，便于排查"为什么搜不到" |
| **P1-6** | **存在性提示开关**：越权剔除时是否提示"存在但你没权限"（默认关闭，见 Q6） |

### P2（可选）

| 编号 | 需求 |
|---|---|
| **P2-1** | **密级变更双人审批**：管理员提/降密级需第二名管理员复核 |
| **P2-2** | **项目维度 UI 管理**：项目的创建、成员维护、到期提醒 |
| **P2-3** | **权限变更影响面预览**：提级前提示"将使 N 人失去可见性" |
| **P2-4** | **定期权限复核（access review）**：定期提醒管理员复核高密级文档的成员清单 |
| **P2-5** | **图片独立 ACL 编辑器**：逐张图片配置 ACL（本期只做提级/剔除） |

---

## 8. 待确认问题（选择题，含推荐项）

> 前三题阻塞 P0 开工，建议优先拍板。

**Q1（阻塞）· 密级档位数量？**
- A. 3 档（内部 / 保密 / 机密）
- **B. 4 档（公开 / 内部 / 保密 / 机密）** ← **推荐**
- C. 5 档（+ 绝密）

推荐理由：4 档足够覆盖企业常见分级，且 `public=0` 让"公司公告/对外材料"这类内容不必再走 need-to-know 例外，减少例外滥用。5 档会让"保密/机密/绝密"的边界在实际标注中模糊，反而提高标注错误率。

**Q2（阻塞）· 存量未标注密级的文档，默认值取多少？**
- A. 取 0（公开）—— 风险最高，等于存量全开
- **B. 取 1（内部）** ← **推荐**
- C. 取 3（机密）—— 存量瞬间全不可见，业务中断
- D. 按现有 `access_level` 推导（private→2，department→1，tenant→0）

推荐理由：B 在安全与可用性之间取平衡，配合 P1-3 的回填工具 + "严格模式"开关（需要更高安全水位时由管理员手动切换为 C），既有默认可用路径，也保留收紧手段。

**Q3（阻塞）· 项目维度如何表达？**
- A. `access_level` 增加第四值 `project`
- **B. 独立字段 `project_ids` + `visibility_mode`（`tier` / `project`），不动 `access_level` 三值** ← **推荐**

推荐理由：① 完全不动现有三层语义与前端标签体系（符合"额外多的部分不要删除"）；② 项目是横向维度，与层级正交 —— 一份文档可以同时"属于部门库"+"属于某项目"，塞进单值枚举会丢失这个组合；③ 存量缺失 `project_ids` 时退回原三层判定，不会 fail-open。

**Q4 · 图片级收紧本期做到什么程度？**
- A. 本期不做（图片一律继承文档）
- **B. 做"提级"与"剔除"两种，不做独立 ACL 编辑器** ← **推荐**
- C. 全做（含逐图 ACL 编辑器）

推荐理由：B 覆盖了"含证件截图"这一最高频场景，改动可控；完整编辑器属于低频需求，放 P2 更稳妥。但**"OCR 派生取严"（P0-8）无论选哪项都必须做** —— 那是最容易被忽略的泄密口。

**Q5 · need-to-know 例外本期是否做？**
- **A. 做，且限定为"对象级显式授予 + 必须带有效期 + 审批链"** ← **推荐**
- B. 不做（本期严格按 clearance 一刀切）

推荐理由：不做的话，"项目组临时需要看一份高密级文档"这类真实诉求会逼着管理员去**整体下调文档密级**，那比开一个带有效期的例外危险得多。例外必须受限，但要有出口。

**Q6 · 越权剔除时，是否提示"存在但你没权限"？**
- **A. 不提示存在性，只说"部分内容因权限限制未包含"** ← **推荐**
- B. 提示文档名（便于用户去申请权限）
- C. 对 kb_admin 及以上提示文档名，其余不提示

推荐理由：B 会造成**存在性泄露**（"某公司密薪文档存在"本身就是信息），且会被批量探测利用。C 是可选折中，但会显著增加实现与测试复杂度，建议二期再说。

**Q7 · 平台管理员 admin 是否豁免密级？**
- **A. 不豁免。admin 默认 clearance=3（有上限），可授予他人密级但不自动拥有** ← **推荐**
- B. admin 豁免所有密级（可见一切）

推荐理由：与现有"admin 也看不到他人个人库"的红线精神一致 —— 本系统从未把 admin 设计成上帝账号。B 会让 admin 账号成为单点泄密风险，且使密级体系对最高权限账号形同虚设。

**Q8 · 级联一致性策略？**
- **A. 收紧同步（走 PG 权威复核）、放宽异步（最终一致）** ← **推荐**
- B. 全部同步（每次变更同步刷新所有派生对象后再返回）

推荐理由：A 在"泄密窗口为零"与"性能可接受"之间取到平衡点 —— 收紧走权威复核本来就没有额外写入成本，放宽走异步不影响安全。B 在万级 chunk 的文档上会造成明显写入延迟，收益却只是"放宽更快可见"。

---

## 9. 验收标准汇总（按需求编号）

| 需求 | 验收方式 | 优先级 |
|---|---|---|
| P0-1 五维 UserScope | 请求入口产出五项属性；链路内不可变；不同 Scope 不共用缓存 | P0 |
| P0-2 密级建模 | `effective_security_level > clearance` 的对象在任何环节均不出现；角色不豁免 | P0 |
| P0-3 项目维度 | 跨部门项目成员可见；同部门非成员不可见；`access_level` 三值未变更 | P0 |
| P0-4 Metadata Schema | PG 与 Qdrant payload 字段名/类型与第 4 节字段表一致，扁平结构 | P0 |
| P0-5 双路下推 | 两路 Scope 表达式序列化逐字节相同；召回前过滤；异常 fail-closed | P0 |
| P0-6 检索后复核 | 制造级联延迟后查询，结果不含已收紧对象；剔除均留审计 | P0 |
| P0-7 继承与级联 | 派生对象只收紧不放宽；`acl_sync_state` 正确标记 | P0 |
| P0-8 图片收紧 + OCR 取严 | 图片提级后，原图不可回显且 OCR 文本/表格/代码不可检索 | P0 |
| P0-9 最终校验 + 溯源 | 全剔除时拒绝回答；部分剔除时部分回答且不泄露存在性；引用点击再校验 | P0 |
| P0-10 越权审计 | 第 7/11/12 环剔除均有可查询的审计记录 | P0 |
| P0-11 测试套件 | A1–A10 全绿 | P0 |
| P1-1 need-to-know | 带有效期的例外生效；过期失效；禁止自我授予 | P1 |
| P1-2 项目临时成员 | 到期自动失效 | P1 |
| P1-3 存量回填 | 未标注清单可导出；批量标注生效；严格模式开关可用 | P1 |
| P1-4 图片自动提级 | 含证件类图片自动提级并可人工复核 | P1 |
| P1-5 Scope 可观测 | 日志含 `scope_fingerprint` 与剔除计数 | P1 |
| P1-6 存在性提示开关 | 开关默认关闭；开启后行为符合预期 | P1 |
