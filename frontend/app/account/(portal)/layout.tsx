import { AccountShell } from "@/components/account/AccountShell";

export default function AccountPortalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <AccountShell>{children}</AccountShell>;
}
