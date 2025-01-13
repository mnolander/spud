import os
from PIL import Image

# Get the input directory full of PNGs
inputdir = input("Enter the location of your input directory (full of PNGs): ")
directory = os.fsencode(inputdir)

# Get the background image
background_file = input("Enter the name of your background image (JPEG): ")
background_path = os.path.abspath(background_file)

# Get the desired output directory
outputdir = input("Enter the location of your output directory: ")
outputdir = os.path.abspath(outputdir)

# Create the output directory if it doesn't exist
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# Define the desired size for the background
desired_size = (4024, 3036)

# Loop through all files in the input directory
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        # Open and resize the background image
        background = Image.open(background_path).convert("RGBA")
        background = background.resize(desired_size)

        # Open the front image
        frontImage = Image.open(os.path.join(inputdir, filename)).convert("RGBA")
        
        # Calculate width and height to center the image
        width = (background.width - frontImage.width) // 2
        height = (background.height - frontImage.height) // 2
        
        # Paste the frontImage onto the resized background
        background.paste(frontImage, (width, height), frontImage)
        
        # Save the output in the specified output directory with a unique filename
        output_filename = os.path.join(outputdir, f"wood_floor_with_carpet_{filename}")
        background.save(output_filename, format="png")

print(f"Images have been processed and saved successfully in '{outputdir}'.")
