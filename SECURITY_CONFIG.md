# EXO Cluster Manager 安全配置指南

## 概述

本次安全修复已解决以下问题：
- ✅ 弱密码哈希算法 → bcrypt/PBKDF2
- ✅ 硬编码管理员密码 → 环境变量
- ✅ 宽松 CORS 配置 → 限制域名白名单
- ✅ 微信密钥明文存储 → 环境变量优先读取
- ✅ Cookie 缺少 Secure 标志 → 自动检测环境
- ✅ 缺少安全响应头 → 添加完整安全头中间件
- ✅ MD5 使用 → SHA-256

---

## 环境变量配置 (生产环境必需)

### 1. 基础安全配置

```bash
# ===== 必需的环境变量 (生产环境) =====

# 运行环境标识 (production/prod 启用严格安全模式)
export EXO_ENV=production

# 强制 HTTPS (启用 Cookie Secure 标志和 HSTS)
export EXO_FORCE_HTTPS=true

# ===== 管理员密码配置 =====
# 方式1: 设置默认管理员密码 (推荐)
export EXO_ADMIN_DEFAULT_PASSWORD="your-secure-password-here"

# 方式2: 首次初始化密码 (仅在首次启动时使用)
export EXO_INIT_PASSWORD="your-init-password"

# ===== CORS 配置 =====
# 允许的前端域名 (逗号分隔)
export EXO_CORS_ORIGINS="https://yourdomain.com,https://admin.yourdomain.com"

# 开发模式: 允许所有来源 (仅用于本地调试!)
# export EXO_DEV_MODE=true
```

### 2. 微信登录配置

```bash
# ===== 微信开放平台 (网站应用) =====
export WECHAT_APP_ID="wx-your-app-id"
export WECHAT_APP_SECRET="your-app-secret"
export WECHAT_REDIRECT_URI="https://yourdomain.com/auth/wechat/callback"
export WECHAT_SCOPE="snsapi_login"
export WECHAT_STATE_PREFIX="exo_"

# ===== 微信小程序 (可选) =====
export WECHAT_MINI_APPID="your-mini-appid"
export WECHAT_MINI_SECRET="your-mini-secret"
```

---

## 快速启动示例

### 开发环境 (.env 文件)

```bash
# .env.development
EXO_ENV=development
EXO_DEV_MODE=true
EXO_ADMIN_DEFAULT_PASSWORD=admin123
WECHAT_APP_ID=wx57fdf2979f1511d1
WECHAT_APP_SECRET=acc6daf9f96e181634b6ef0005c6ceee
```

启动命令:
```bash
# 加载 .env 文件并启动
set -a && source .env.development && set -a
python server.py --port 8080
```

### 生产环境 (.env 文件)

```bash
# .env.production
EXO_ENV=production
EXO_FORCE_HTTPS=true
EXO_ADMIN_DEFAULT_PASSWORD=${SECURE_ADMIN_PASSWORD}  # 从 secrets manager 获取
EXO_CORS_ORIGINS=https://yourdomain.com,https://admin.yourdomain.com

# 微信配置 (必须从环境变量读取，不支持配置文件!)
export WECHAT_APP_ID=${WECHAT_APP_ID_FROM_SECRET}
export WECHAT_APP_SECRET=${WECHAT_APP_SECRET_FROM_SECRET}
```

Docker 示例:
```dockerfile
# Dockerfile
ENV EXO_ENV=production
ENV EXO_FORCE_HTTPS=true
ENV EXO_ADMIN_DEFAULT_PASSWORD=""
ENV EXO_CORS_ORIGINS="https://yourdomain.com"

# 通过 docker-secrets 或 Kubernetes Secrets 注入敏感信息
```

---

## 安装依赖

### bcrypt (推荐)

```bash
pip install bcrypt
```

**如果未安装 bcrypt**, 系统将自动回退到 PBKDF2-SHA256 (100,000 次迭代)，安全性仍然可接受。

---

## 安全特性详解

### 1. 密码哈希改进

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| **算法** | SHA-256 + 静态盐 | bcrypt (工作因子 12) 或 PBKDF2 |
| **盐值** | 所有用户相同 (`exo_cluster_salt_v1`) | 每个密码随机生成 |
| **抗暴力破解** | 弱 (SHA-256 速度快) | 强 (bcrypt ~250ms/次) |
| **向后兼容** | - | 支持旧格式迁移 |

**旧密码迁移**: 
- 旧格式的密码哈希仍可验证 (带警告日志)
- 建议用户下次登录时修改密码以升级为新格式

### 2. Cookie 安全

| 环境 | Secure | HttpOnly | SameSite |
|------|--------|----------|----------|
| **生产环境** | ✅ 强制启用 | ✅ 始终启用 | lax |
| **开发环境** | ❌ 允许 HTTP | ✅ 始终启用 | lax |

### 3. CORS 策略

| 模式 | allow_origins | allow_credentials | 适用场景 |
|------|---------------|-------------------|----------|
| **安全模式** | 白名单域名 | true | 生产环境 |
| **开发模式** | * (全部) | false | 本地调试 |

### 4. 安全响应头

自动添加以下 HTTP 头:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=(), ...
Content-Security-Policy: ... (根据环境自动调整)
Strict-Transport-Security: max-age=31536000; includeSubDomains (仅生产)
```

### 5. 微信密钥管理

**优先级顺序**:
1. ✅ 环境变量 `WECHAT_APP_ID` / `WECHAT_APP_SECRET` (推荐)
2. ⚠️ 配置文件 `wechat_config.json` (仅开发，生产环境会拒绝)

**生产环境保护**:
- 如果检测到 `EXO_ENV=production` 且从文件读取密钥，系统会拒绝启动并记录错误日志

---

## 验证安全配置

### 1. 检查响应头

```bash
curl -I https://yourdomain.com/api/nodes

# 应包含以下头:
# X-Content-Type-Options: nosniff
# X-Frame-Options: DENY
# Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### 2. 测试 CORS

```bash
# 测试跨域请求 (应被拒绝或返回正确的 Access-Control-Allow-Origin)
curl -H "Origin: https://evil.com" \
     -H "Access-Control-Request-Method: POST" \
     -X OPTIONS \
     https://yourdomain.com/v1/chat/completions
```

### 3. 验证 Cookie 安全性

```bash
# 登录后检查 Set-Cookie 头:
curl -v -c cookies.txt -X POST \
     https://yourdomain.com/login/email \
     -H "Content-Type: application/json" \
     -d '{"email":"test@example.com","password":"xxx"}'

# 应包含: Secure; HttpOnly; SameSite=Lax
```

---

## 故障排除

### 问题: bcrypt 未安装

**症状**:
```
WARNING: bcrypt 未安装，将使用 PBKDF2 作为后备方案
```

**解决方案**:
```bash
pip install bcrypt
```

### 问题: 生产环境微信登录失败

**症状**:
```
ERROR: 生产环境检测到从配置文件读取微信密钥，请改用环境变量!
```

**原因**: 在 `EXO_ENV=production` 下尝试从 `wechat_config.json` 读取密钥

**解决方案**:
设置环境变量:
```bash
export WECHAT_APP_ID="your-app-id"
export WECHAT_APP_SECRET="your-app-secret"
```

### 问题: CORS 错误

**症状**: 前端控制台报跨域错误

**检查清单**:
1. 确认 `EXO_CORS_ORIGINS` 包含前端域名
2. 如需调试，临时设置 `EXO_DEV_MODE=true`
3. 生产环境不要使用 `EXO_DEV_MODE=true`

---

## 后续建议

### 高优先级
- [ ] 将 `users.json` 和 `api_keys.json` 迁移至数据库
- [ ] 实现 API 速率限制中间件
- [ ] 添加操作审计日志

### 中优先级
- [ ] 集成 2FA/MFA 双因素认证
- [ ] 实现密码过期策略
- [ ] 添加 IP 白名单功能

### 低优先级
- [ ] 集成 WAF (Web Application Firewall)
- [ ] 定期依赖安全扫描 (dependabot/snyk)
- [ ] 渗透测试和安全评估

---

## 参考链接

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CORS 安全最佳实践](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)
- [Cookie 安全属性](https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies)
- [Content Security Policy](https://developer.mozilla.org/en-US/docs/Web/HTTP/CSP)

---

**最后更新**: 2026-06-12  
**安全审计版本**: v1.0-security-fix
