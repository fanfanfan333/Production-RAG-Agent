import { apiFetch } from "@/lib/api/client";

/**
 * Authentication-related account APIs (修改密码).
 * Login/register live in the auth context — they manage session state.
 */

export interface ChangePasswordResult {
  changed: boolean;
  message: string;
}

export async function changePassword(
  oldPassword: string,
  newPassword: string
): Promise<ChangePasswordResult> {
  return apiFetch<ChangePasswordResult>("/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
  });
}
