"use client";

import * as React from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * 轻量原生下拉框.
 *
 * 为什么不用 Radix Select：企业后台里下拉只需要"选一个受控值"，原生
 * ``<select>`` 已经具备键盘、无障碍、移动端原生选择器的全部行为，还省掉
 * 一个依赖与一层 Portal。样式上对齐 Input 的规格（h-9、圆角、边框变量）。
 */
function Select({ className, children, ...props }: React.ComponentProps<"select">) {
  return (
    <div className="relative">
      <select
        className={cn(
          "flex h-9 w-full appearance-none rounded-md border border-input bg-transparent px-3 py-1 pr-8",
          "text-[13px] shadow-sm transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
          "disabled:cursor-not-allowed disabled:opacity-50",
          className
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute top-1/2 right-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
    </div>
  );
}

function SelectItem({ ...props }: React.ComponentProps<"option">) {
  return <option {...props} />;
}

export { Select, SelectItem };
