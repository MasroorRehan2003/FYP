import os



# Base directory

base_dir = "clothing_images"



# Clothing categories predicted by your model

predicted_categories = [
    "kurtas",
    "shirts",
    "tops",
    "jeans",
    "trousers",
    "leggings",
    "capris",
    "jacket",
    "coat",
    "puffer jacket",
    "dupata",
    "none"  # This is to handle cases where no outerwear is needed
]



# Create the base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


# Create subdirectories for each category
for category in predicted_categories:
    folder_name = category.replace(" ", "_").lower()  # handle spaces
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)



print("All directories created successfully inside 'clothing_images/' folder.")
