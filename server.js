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
