export const foregroundFraction = (pixels, channels, background) => {
  let foreground = 0;
  for (let offset = 0; offset < pixels.length; offset += channels) {
    const difference = Math.max(
      Math.abs(pixels[offset] - background[0]),
      Math.abs(pixels[offset + 1] - background[1]),
      Math.abs(pixels[offset + 2] - background[2]),
    );
    if (difference > 16) foreground++;
  }
  return foreground / (pixels.length / channels);
};
