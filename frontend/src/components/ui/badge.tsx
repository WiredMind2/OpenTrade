import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded px-2 py-0.5 text-xs font-medium transition-tv",
  {
    variants: {
      variant: {
        default:
          "bg-primary/10 text-primary border border-primary/20",
        secondary:
          "bg-secondary text-secondary-foreground border border-tv-border-secondary",
        destructive:
          "bg-destructive/10 text-destructive border border-destructive/20",
        outline: "text-foreground border border-tv-border-secondary",
        success:
          "bg-tv-green/10 text-tv-green border border-tv-green/20",
        warning:
          "bg-tv-orange/10 text-tv-orange border border-tv-orange/20",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
