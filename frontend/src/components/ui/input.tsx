import * as React from "react"

import { cn } from "@/lib/utils"

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={cn(
          "flex h-8 w-full rounded border border-tv-border-secondary bg-tv-bg-tertiary px-3 py-1.5 text-sm text-tv-text-primary transition-tv file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-tv-text-tertiary focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary focus-visible:border-primary hover:border-tv-border-hover disabled:cursor-not-allowed disabled:opacity-40",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Input.displayName = "Input"

export { Input }