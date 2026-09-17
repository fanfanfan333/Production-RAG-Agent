import type { NextConfig } from "next";

const backendUrl =
  process.env.BACKEND_URL?.replace(/\/$/, "") || "http://localhost:8000";

const nextConfig: NextConfig = {
  // 构建与运行时缓存全部落 D 盘，避免 Next.js 的 .next 缓存给 C 盘造成压力
  // （.next 默认在项目内，这里显式指到 D 盘依赖目录；项目本身已在 D 盘）
  distDir: process.env.NEXT_DIST_DIR || ".next",

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
