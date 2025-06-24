import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from PIL import Image
import textwrap

# ---------- CONFIGURATION ---------- #
base_path = os.path.join(os.getcwd(), "dataset")
train_csv = os.path.join(base_path, "Training_set.csv")
train_img_folder = os.path.join(base_path, "train")

# ---------- LOAD DATA ---------- #
train_df = pd.read_csv(train_csv)
train_df['file_path'] = train_df['filename'].apply(lambda x: os.path.join(train_img_folder, x))

# Optional: Shorten long species names for display
def shorten_name(name, width=25):
    return "\n".join(textwrap.wrap(name, width=width))

train_df['short_label'] = train_df['label'].apply(lambda x: shorten_name(x))

# ---------- 1. CLEAN BAR CHART ---------- #
plt.figure(figsize=(12, 10))
sns.countplot(y='short_label', data=train_df, order=train_df['short_label'].value_counts().index, palette='Spectral')
plt.title("📊 Number of Images per Butterfly Species", fontsize=18)
plt.xlabel("Image Count", fontsize=12)
plt.ylabel("Species", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.show()

# ---------- 2. CLEAN PIE CHART ---------- #
species_counts = train_df['label'].value_counts()
top10 = species_counts.head(10)
labels = [shorten_name(name, width=20) for name in top10.index]

plt.figure(figsize=(8, 8))
plt.pie(top10, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
plt.title("🧁 Top 10 Butterfly Species Distribution", fontsize=16)
plt.axis('equal')
plt.tight_layout()
plt.show()

# ---------- 3. CLEAN SAMPLE IMAGE GRID ---------- #
top_species = train_df['label'].value_counts().index[:6]
plt.figure(figsize=(18, 10))
for i, species in enumerate(top_species):
    img_path = train_df[train_df['label'] == species].iloc[0]['file_path']
    img = mpimg.imread(img_path)
    plt.subplot(2, 3, i + 1)
    plt.imshow(img)
    plt.title(shorten_name(species), fontsize=12)
    plt.axis('off')

plt.suptitle("🦋 Sample Images from Top 6 Butterfly Species", fontsize=20)
plt.tight_layout()
plt.show()

# ---------- 4. IMAGE SIZE DISTRIBUTION ---------- #
widths = []
heights = []
for path in train_df['file_path'][:500]:
    try:
        with Image.open(path) as img:
            width, height = img.size
            widths.append(width)
            heights.append(height)
    except:
        continue

plt.figure(figsize=(10, 6))
sns.histplot(widths, color='skyblue', label='Width', kde=True)
sns.histplot(heights, color='salmon', label='Height', kde=True)
plt.title("📏 Image Width and Height Distribution", fontsize=16)
plt.xlabel("Pixels", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()
