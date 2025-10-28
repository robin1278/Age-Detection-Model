from PIL import Image, ImageDraw, ImageFont

# Create a blank image
width, height = 640, 480
image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

# Add text to the image
try:
    font = ImageFont.truetype("arial.ttf", 40)
except IOError:
    font = ImageFont.load_default()

text = "Placeholder Image"
bbox = draw.textbbox((0, 0), text, font=font)
textwidth = bbox[2] - bbox[0]
textheight = bbox[3] - bbox[1]
x = (width - textwidth) / 2
y = (height - textheight) / 2
draw.text((x, y), text, font=font, fill="black")

# Save the image
image.save("image_enhancement/placeholder.jpg")
