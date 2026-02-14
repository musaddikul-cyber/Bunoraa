import * as React from "react";
import { cn } from "@/lib/utils";

type FileDropzoneProps = {
  label: string;
  description?: string;
  accept?: string;
  multiple?: boolean;
  maxFiles?: number;
  value: File[];
  onChange: (files: File[]) => void;
};

export function FileDropzone({
  label,
  description,
  accept,
  multiple = false,
  maxFiles = 5,
  value,
  onChange,
}: FileDropzoneProps) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const [dragActive, setDragActive] = React.useState(false);
  const inputId = React.useId();

  const handleFiles = (files: FileList | null) => {
    if (!files) return;
    const incoming = Array.from(files);
    const merged = multiple ? [...value, ...incoming] : incoming.slice(0, 1);
    onChange(merged.slice(0, maxFiles));
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold">{label}</p>
        <button
          type="button"
          className="text-xs font-semibold text-primary"
          onClick={() => inputRef.current?.click()}
        >
          Browse files
        </button>
      </div>
      {description ? (
        <p className="text-xs text-foreground/60">{description}</p>
      ) : null}
      <div
        role="button"
        tabIndex={0}
        aria-label={label}
        className={cn(
          "min-h-28 rounded-xl border border-dashed px-4 py-6 text-center text-sm text-foreground/70 transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          dragActive ? "border-primary bg-primary/5" : "border-border bg-muted/30"
        )}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragActive(false);
          handleFiles(event.dataTransfer.files);
        }}
      >
        <p>Drag and drop files here, or tap to browse.</p>
        <p className="mt-2 text-xs text-foreground/50">
          {multiple ? `Up to ${maxFiles} files` : "Single file"}
        </p>
      </div>
      <input
        id={inputId}
        ref={inputRef}
        type="file"
        accept={accept}
        multiple={multiple}
        className="hidden"
        onChange={(event) => handleFiles(event.target.files)}
      />
      {value.length ? (
        <div className="space-y-2 rounded-xl border border-border bg-card p-3 text-xs">
          {value.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex items-center justify-between gap-2"
            >
              <span className="truncate pr-2">{file.name}</span>
              <button
                type="button"
                className="rounded-md px-2 py-1 text-primary hover:bg-primary/10"
                onClick={() => {
                  const next = [...value];
                  next.splice(index, 1);
                  onChange(next);
                }}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
