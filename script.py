import os

# List of faulty TIFF-mislabeled files
bad_files = [
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_10_10.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_10_27.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_11_30.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_11_6.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_12_15.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_12_34.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_13_2.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_13_22.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_14_11.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_14_19.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_15_35.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_18_23.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_20_6.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_26_10.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_27_11.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_34_6.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_39_1.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_42_1.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_43_9.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_46_14.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_56_1.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_5_9.jpg",
    r"C:\Users\Admin\Desktop\Visionxaid\Dataset\train\AMD\aria_a_9_43.jpg"
]

for file in bad_files:
    try:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️ Deleted: {file}")
        else:
            print(f"⚠️ File not found: {file}")
    except Exception as e:
        print(f"❌ Could not delete {file}: {e}")