import { FlatCompat } from "@eslint/eslintrc";
import { fileURLToPath } from "node:url";

const compat = new FlatCompat({
  baseDirectory: fileURLToPath(new URL(".", import.meta.url)),
});

export default [
  ...compat.extends("next/core-web-vitals"),
  { ignores: ["lib/api-client/schema.d.ts"] },
];
