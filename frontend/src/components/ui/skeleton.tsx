import { cn } from "@/lib/utils"

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("shimmer rounded bg-tv-bg-tertiary", className)}
      {...props}
    />
  )
}

export { Skeleton }
