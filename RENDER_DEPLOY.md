# Render 部署指南

## 概述

从 Railway 迁移到 Render 免费计划。Render 提供 750 小时/月的免费 Web Service。

## 部署步骤

### 1. 推送代码到 GitHub

确保所有文件已提交并推送到 GitHub 仓库，包括：
- `Dockerfile`
- `render.yaml`
- `.dockerignore`

### 2. 在 Render 创建服务

1. 访问 https://dashboard.render.com
2. 点击 **New +** → **Web Service**
3. 选择 **Build and deploy from a Git repository**
4. 连接你的 GitHub 仓库
5. Render 会自动检测到 `render.yaml`，确认配置：
   - **Runtime**: Docker
   - **Plan**: Free
   - **Dockerfile Path**: `./Dockerfile`
6. 在 **Environment** 中添加密钥：
   - `TUSHARE_TOKEN` = 你的 tushare token
   - `UPSTASH_REDIS_REST_URL` = （可选）
   - `UPSTASH_REDIS_REST_TOKEN` = （可选）
7. 点击 **Create Web Service**

### 3. 更新 Cloudflare Worker

部署成功后，Render 会分配一个 URL，格式如：
`https://stock-prediction-api-xxxx.onrender.com`

更新 `wrangler.toml` 中的 `PREDICTION_API_URL`：

```toml
[vars]
PREDICTION_API_URL = "https://stock-prediction-api-xxxx.onrender.com"
```

然后重新部署 Worker：

```bash
cd /root/workspace/quarnt
npx wrangler deploy
```

### 4. 验证

```bash
# 测试 Render 后端
curl https://stock-prediction-api-xxxx.onrender.com/health

# 测试 Cloudflare Worker 代理
curl https://your-worker.workers.dev/health

# 测试预测
curl "https://your-worker.workers.dev/predict?stock=000001.SZ&fast_mode=true"
```

## 注意事项

- **冷启动**：免费计划 15 分钟无请求会休眠，首次请求需 30-60 秒唤醒
- **内存**：512MB RAM，对 CatBoost 推理足够
- **CPU**：共享 CPU，预测可能较慢，建议用 `fast_mode=true` 跳过训练
- **磁盘**：无持久化存储，重启后模型/缓存丢失
- **每月限额**：750 小时（约 31 天），足够 24/7 运行一个服务
