export function getLazyImageProps(
  src: string,
  alt: string,
  additionalProps?: React.ImgHTMLAttributes<HTMLImageElement>
) {
  return {
    src,
    alt,
    loading: "lazy" as const,
    decoding: "async" as const,
    ...additionalProps,
  };
}
