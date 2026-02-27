#!/bin/bash
# 生产环境极速启动脚本 (Node.js直接模式)

set -e

echo "🚀 开始生产环境部署 (Node.js直接模式)..."

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查Node.js
if ! command -v node > /dev/null 2>&1; then
    echo -e "${RED}❌ Node.js未安装${NC}"
    exit 1
fi

NODE_VERSION=$(node --version)
echo -e "${GREEN}✓ Node.js版本: $NODE_VERSION${NC}"

# 创建生产环境变量文件
if [ ! -f .env.production ]; then
    echo -e "${YELLOW}⚠️ 创建生产环境变量文件...${NC}"
    cat > .env.production << 'EOF'
# 生产环境配置
NODE_ENV=production
PORT=3000
HOST=0.0.0.0
LOG_LEVEL=info
API_KEY=prod-api-key-20260227
EOF
fi

# 加载环境变量
set -a
source .env.production 2>/dev/null || true
set +a

# 创建健康检查脚本
cat > healthcheck.js << 'EOF'
const http = require('http');
const port = process.env.PORT || 3000;

const options = {
  hostname: 'localhost',
  port: port,
  path: '/health',
  method: 'GET',
  timeout: 2000
};

const req = http.request(options, (res) => {
  if (res.statusCode === 200) {
    console.log('✓ Health check passed');
    process.exit(0);
  } else {
    console.log(`✗ Health check failed with status: ${res.statusCode}`);
    process.exit(1);
  }
});

req.on('error', (err) => {
  console.log(`✗ Health check error: ${err.message}`);
  process.exit(1);
});

req.on('timeout', () => {
  console.log('✗ Health check timeout');
  req.destroy();
  process.exit(1);
});

req.end();
EOF

# 创建生产级服务器
cat > server.js << 'EOF'
const http = require('http');
const cluster = require('cluster');
const os = require('os');

const port = process.env.PORT || 3000;
const host = process.env.HOST || '0.0.0.0';

// 请求计数器
let requestCount = 0;
let startTime = Date.now();

// 创建服务器
const server = http.createServer((req, res) => {
  requestCount++;
  
  // 设置安全响应头
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');
  
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ 
      status: 'healthy', 
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      pid: process.pid,
      requests: requestCount
    }));
  } else if (req.url === '/metrics') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      cpu: process.cpuUsage(),
      requests: requestCount,
      startTime: new Date(startTime).toISOString()
    }));
  } else if (req.url === '/ready') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ready: true }));
  } else {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ 
      message: 'Production server running',
      env: process.env.NODE_ENV,
      version: '1.0.0',
      pid: process.pid,
      timestamp: new Date().toISOString()
    }));
  }
});

server.listen(port, host, () => {
  console.log(`🚀 Production server running on http://${host}:${port}`);
  console.log(`📊 PID: ${process.pid}`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`);
});

// 优雅关闭
process.on('SIGTERM', () => {
  console.log('SIGTERM received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('SIGINT received, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});
EOF

# 创建package.json
if [ ! -f package.json ]; then
    cat > package.json << 'EOF'
{
  "name": "production-app",
  "version": "1.0.0",
  "description": "Production ready application",
  "main": "server.js",
  "scripts": {
    "start": "NODE_ENV=production node server.js",
    "health": "node healthcheck.js",
    "dev": "node server.js"
  },
  "engines": {
    "node": ">=22.0.0"
  },
  "keywords": ["production", "api"],
  "author": "",
  "license": "MIT"
}
EOF
fi

echo -e "${GREEN}✓ 基础文件创建完成${NC}"

# 停止已存在的服务
if pgrep -f "node server.js" > /dev/null; then
    echo -e "${YELLOW}⚠️ 停止已存在的服务...${NC}"
    pkill -f "node server.js" || true
    sleep 2
fi

# 启动服务
echo -e "${YELLOW}🚀 启动生产服务...${NC}"
export NODE_ENV=production
export PORT=3000
nohup node server.js > server.log 2>&1 &
echo $! > server.pid

# 等待服务启动
echo -e "${YELLOW}⏳ 等待服务就绪 (3秒)...${NC}"
sleep 3

# 健康检查
echo -e "${YELLOW}🏥 执行健康检查...${NC}"
if [ -f server.pid ]; then
    PID=$(cat server.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 服务进程运行正常 (PID: $PID)${NC}"
        
        # 测试健康端点
        if node healthcheck.js; then
            echo ""
            echo -e "${GREEN}🎉 生产环境部署成功！${NC}"
            echo ""
            echo -e "${BLUE}📊 服务状态:${NC}"
            echo "  PID: $PID"
            echo "  端口: $PORT"
            echo "  环境: $NODE_ENV"
            echo ""
            echo -e "${BLUE}🔗 访问地址:${NC}"
            echo "  - 首页:     http://localhost:$PORT"
            echo "  - 健康检查: http://localhost:$PORT/health"
            echo "  - 就绪检查: http://localhost:$PORT/ready"
            echo "  - 指标:     http://localhost:$PORT/metrics"
            echo ""
            echo -e "${BLUE}📋 常用命令:${NC}"
            echo "  - 查看日志: tail -f server.log"
            echo "  - 停止服务: kill \$(cat server.pid)"
            echo "  - 健康检查: node healthcheck.js"
            echo ""
            echo -e "${BLUE}📁 生成的文件:${NC}"
            ls -la *.js *.json .env.production 2>/dev/null | awk '{print "  " $9}'
        else
            echo -e "${RED}❌ 健康检查未通过${NC}"
            echo "日志:"
            tail -20 server.log
            exit 1
        fi
    else
        echo -e "${RED}❌ 服务进程未运行${NC}"
        echo "日志:"
        tail -20 server.log
        exit 1
    fi
else
    echo -e "${RED}❌ PID文件不存在${NC}"
    exit 1
fi
