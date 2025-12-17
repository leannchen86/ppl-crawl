"""
Human Baseline Test for Face-Name Association
Shows random face pairs and asks you to guess which name matches.

Usage:
    python human_baseline_test.py --names david michael --trials 20

Literature baseline: ~60-70% accuracy for same-gender face-name matching
(Zwebner et al., 2017 - "We Look Like Our Names")
"""
import argparse
import json
import random
import os
from PIL import Image
import matplotlib.pyplot as plt


def load_images_for_name(name: str, index_dir: str, num_samples: int = 50):
    """Load random good images for a name."""
    index_path = os.path.join(index_dir, f"index_{name}.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    
    with open(index_path) as f:
        index = json.load(f)
    
    good_images = index.get("good", [])
    if len(good_images) < num_samples:
        num_samples = len(good_images)
    
    return random.sample(good_images, num_samples)


def run_test(names: list, index_dir: str, num_trials: int):
    """Run the human baseline test."""
    print("\n" + "="*60)
    print("HUMAN BASELINE TEST: Face-Name Association")
    print("="*60)
    print(f"\nNames: {names}")
    print(f"Trials: {num_trials}")
    print(f"\nLiterature baseline: ~60-70% for same-gender pairs")
    print("(Zwebner et al., 2017 - 'We Look Like Our Names')")
    print("\n" + "-"*60)
    print("Instructions:")
    print("1. Look at the face shown")
    print("2. Type the number corresponding to your guess")
    print("3. Press Enter to submit")
    print("-"*60 + "\n")
    
    # Load images for each name
    name_images = {}
    for name in names:
        name_images[name] = load_images_for_name(name, index_dir)
    
    correct = 0
    total = 0
    
    for trial in range(num_trials):
        # Pick a random name and image
        true_name = random.choice(names)
        img_path = random.choice(name_images[true_name])
        
        # Show image
        try:
            img = Image.open(img_path)
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Trial {trial + 1}/{num_trials}: Which name?")
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue
        
        # Get user guess
        print(f"\nTrial {trial + 1}/{num_trials}")
        for i, name in enumerate(names):
            print(f"  {i + 1}. {name.capitalize()}")
        
        while True:
            try:
                guess_idx = int(input("Your guess (number): ")) - 1
                if 0 <= guess_idx < len(names):
                    break
                print("Invalid number, try again.")
            except ValueError:
                print("Please enter a number.")
        
        plt.close()
        
        guessed_name = names[guess_idx]
        is_correct = guessed_name == true_name
        
        if is_correct:
            correct += 1
            print(f"✓ Correct! It was {true_name.capitalize()}")
        else:
            print(f"✗ Wrong. It was {true_name.capitalize()}, you guessed {guessed_name.capitalize()}")
        
        total += 1
        print(f"Running accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Final results
    accuracy = correct / total if total > 0 else 0
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Your accuracy: {correct}/{total} = {100*accuracy:.1f}%")
    print(f"Random baseline: {100/len(names):.1f}%")
    print(f"Literature baseline: ~60-70%")
    print()
    
    if accuracy > 0.7:
        print("→ You performed above literature baseline!")
        print("  The task might be learnable with the right approach.")
    elif accuracy > 0.55:
        print("→ You performed near literature baseline (~60-70%).")
        print("  This suggests ~60% model accuracy is reasonable.")
    else:
        print("→ You performed near random chance.")
        print("  The task might be very hard or the data is noisy.")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Human baseline test for face-name association")
    parser.add_argument("--names", nargs="+", default=["david", "michael"],
                        help="Names to test (default: david michael)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of trials (default: 20)")
    parser.add_argument("--index-dir", default="/home/leann/face-detection",
                        help="Directory containing index_*.json files")
    args = parser.parse_args()
    
    run_test(args.names, args.index_dir, args.trials)


if __name__ == "__main__":
    main()

