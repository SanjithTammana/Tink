// Theme-aware artwork: both versions are in the page and CSS shows the one for the current theme.
export default function Art({ name, className = '', alt = '', ...rest }) {
  return (
    <>
      <img className={`art for-light ${className}`} src={`/art/${name}-light.webp`} alt={alt} draggable="false" {...rest} />
      <img className={`art for-dark ${className}`} src={`/art/${name}-dark.webp`} alt={alt} draggable="false" {...rest} />
    </>
  );
}
