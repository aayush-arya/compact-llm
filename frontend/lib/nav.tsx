import {
  BarChart3,
  Columns2,
  Database,
  LayoutDashboard,
  History as HistoryIcon,
  Target,
} from "lucide-react";

export type NavItem = {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
};

export const NAV_GROUPS: { title: string; items: NavItem[] }[] = [
  { title: "Platform", items: [{ href: "/", label: "Overview", icon: LayoutDashboard }] },
  {
    title: "Analyze",
    items: [
      { href: "/score", label: "Score", icon: Target },
      { href: "/playground", label: "Playground", icon: Columns2 },
    ],
  },
  {
    title: "Analysis",
    items: [
      { href: "/evaluation", label: "Evaluation", icon: BarChart3 },
      { href: "/datasets", label: "Datasets", icon: Database },
      { href: "/history", label: "History", icon: HistoryIcon },
    ],
  },
];

export const NAV_ITEMS: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);

export function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname.startsWith(href);
}
