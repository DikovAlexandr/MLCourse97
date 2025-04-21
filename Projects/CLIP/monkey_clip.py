import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay,
    f1_score
)
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import random
import argparse

'''
Mantled howler — Колумбийский ревун
Patas monkey — Мартышка-гусар, или обыкновенный гусар
Bald uakari — Лысый уакари
Japanese macaque — Японский макак
Pygmy marmoset — Карликовая игрунка
White-headed capuchin — Белолобый капуцин
Silvery marmoset — Серебристая игрунка
Common squirrel monkey — Саймири, или Беличьи обезьяны
Black-headed night monkey — Черноголовая мирикина
Nilgiri langur — Капюшонный гульман
'''

# Конфигурация
CONFIG = {
    'data_folders': {
        'train': 'monkeys',
        'validation': 'monkey_validation',
        'people': 'people'
    },
    'results_dir': 'results',
    'classes': [
        "mantled howler", "patas monkey", "bald uakari",
        "japanese macaque", "pygmy marmoset", "white headed capuchin",
        "silvery marmoset", "common squirrel monkey",
        "black headed night monkey", "nilgiri langur"
    ],
    'model_name': 'openai/clip-vit-base-patch32',
    'num_examples': 100,
    'examples_per_class': 10
}

# Инициализация delta_look_like как глобальной переменной
delta_look_like = 0.08

# Инициализация модели
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(CONFIG['model_name']).to(device)
processor = CLIPProcessor.from_pretrained(CONFIG['model_name'])

def load_dataset(root_folder):
    """Загрузка изображений из структурированных папок"""
    image_paths = []
    true_labels = []
    
    for folder in sorted(os.listdir(root_folder)):
        if not folder.startswith('n'):
            continue
            
        try:
            class_idx = int(folder[1:])
            if class_idx >= len(CONFIG['classes']):
                continue
        except ValueError:
            continue
            
        folder_path = os.path.join(root_folder, folder)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_paths.append(os.path.join(folder_path, file))
                true_labels.append(class_idx)
    
    return image_paths, true_labels

def classify_images(image_paths, is_people=False):
    """Классификация изображений с помощью CLIP"""
    results = []
    
    for path in tqdm(image_paths, desc="Classifying images"):
        try:
            image = Image.open(path)
            inputs = processor(
                text=[f"a photo of a {c}" for c in CONFIG['classes']],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
                
            pred_class = np.argmax(probs)
            result = {
                'image_path': path,
                'pred_class': pred_class,
                'pred_label': CONFIG['classes'][pred_class],
                'probability': probs[pred_class],
                'all_probs': probs
            }
            
            if not is_people:
                result['true_class'] = int(os.path.basename(os.path.dirname(path))[1:])
                
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    return pd.DataFrame(results)

def save_metrics(df, dataset_type, results_dir):
    """Сохранение метрик качества"""
    report = classification_report(
        df['true_class'], 
        df['pred_class'],
        target_names=CONFIG['classes'],
        output_dict=True
    )
    
    report_path = os.path.join(results_dir, dataset_type, 'metrics.txt')
    with open(report_path, 'w') as f:
        f.write(classification_report(df['true_class'], df['pred_class'], target_names=CONFIG['classes']))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(df['true_class'], df['pred_class'])))
    
    plt.figure(figsize=(15,12))
    cm = confusion_matrix(df['true_class'], df['pred_class'])
    disp = ConfusionMatrixDisplay(cm, display_labels=CONFIG['classes'])
    disp.plot(xticks_rotation=90, values_format='.0f', cmap='Blues')
    plt.title(f'Confusion Matrix ({dataset_type.capitalize()})')
    cm_path = os.path.join(results_dir, dataset_type, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    
    return report

def save_examples(df, dataset_type, results_dir):
    """Сохранение примеров классификации"""
    examples_dir = os.path.join(results_dir, dataset_type, 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    correct_df = df
    examples = []
    
    for class_idx in range(len(CONFIG['classes'])):
        class_examples = correct_df[correct_df['true_class'] == class_idx]
        if len(class_examples) > 0:
            sample_size = min(CONFIG['examples_per_class'], len(class_examples))
            sampled = class_examples.sample(sample_size)
            examples.extend(sampled.to_dict('records'))
    
    while len(examples) < CONFIG['num_examples'] and len(df) > len(examples):
        new_example = df.sample(1).iloc[0].to_dict()
        if new_example not in examples:
            examples.append(new_example)
    
    for idx, example in enumerate(examples[:CONFIG['num_examples']]):
        fig = plt.figure(figsize=(20, 6))
        
        ax1 = plt.subplot(1, 3, 1)
        img = Image.open(example['image_path'])
        ax1.imshow(img)
        ax1.set_title(
            f"True: {CONFIG['classes'][example['true_class']]}\n"
            f"Pred: {example['pred_label']} ({example['probability']:.2f})"
        )
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 3, 2)
        ax2.barh(CONFIG['classes'], example['all_probs'])
        ax2.set_xlim(0, 1)
        ax2.set_title('Class Probabilities')
        
        ax3 = plt.subplot(1, 3, 3)
        class_folder = os.path.join(
            CONFIG['data_folders']['train'], 
            f"n{example['pred_class']}"
        )
        if os.path.exists(class_folder):
            images = [f for f in os.listdir(class_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if images:
                random_image = random.choice(images)
                monkey_img = Image.open(os.path.join(class_folder, random_image))
                ax3.imshow(monkey_img)
                ax3.set_title(f"Пример {example['pred_label']}")
                ax3.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(examples_dir, f'example_{idx}.png'), bbox_inches='tight')
        plt.close()

def process_people():
    """Обработка фотографий людей с улучшенным отображением обезьян"""
    people_folder = CONFIG['data_folders']['people']
    if not os.path.exists(people_folder):
        return
    
    print("\nProcessing people with improved visualization...")
    people_dir = os.path.join(CONFIG['results_dir'], 'people')
    examples_dir = os.path.join(people_dir, 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    people_images = [
        os.path.join(people_folder, f) 
        for f in os.listdir(people_folder) 
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ]
    
    if not people_images:
        return
    
    people_df = classify_images(people_images, is_people=True)
    
    monkey_cache = {}
    for class_idx in range(len(CONFIG['classes'])):
        class_folder = os.path.join(CONFIG['data_folders']['train'], f"n{class_idx}")
        if os.path.exists(class_folder):
            monkey_cache[class_idx] = [
                os.path.join(class_folder, f) 
                for f in os.listdir(class_folder) 
                if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ]

    for idx, row in people_df.iterrows():
        fig = plt.figure(figsize=(25, 10))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 2, 3], wspace=0.4)
        
        ax1 = fig.add_subplot(gs[0])
        human_img = Image.open(row['image_path'])
        ax1.imshow(human_img)
        ax1.set_title(os.path.basename(row['image_path']), fontsize=12)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[1])
        probs = row['all_probs']
        colors = ['#2ecc71' if p >= delta_look_like else '#95a5a6' for p in probs]
        ax2.barh(CONFIG['classes'], probs, color=colors, height=0.8)
        ax2.axvline(x=delta_look_like, color='#e74c3c', linestyle='--', linewidth=1)
        ax2.set_xlim(0, 1)
        ax2.set_title('Гистограмма похожести на бибизян', fontsize=14)
        ax2.tick_params(axis='y', labelsize=10)
        
        ax3 = fig.add_subplot(gs[2])
        ax3.axis('off')
        
        top_classes = sorted(
            [(i, prob) for i, prob in enumerate(probs) if prob >= delta_look_like],
            key=lambda x: x[1], 
            reverse=True
        )[:4]

        if not top_classes:
            ax3.text(0.5, 0.5, ('Нет классов с вероятностью >' + str(delta_look_like*100) + '%'), 
                    ha='center', va='center', fontsize=14)
        else:
            n_cols = 2
            n_rows = (len(top_classes) + 1) // n_cols
            inner_gs = fig.add_gridspec(
                nrows=n_rows,
                ncols=n_cols,
                left=ax3.get_position().x0,
                right=ax3.get_position().x1,
                top=ax3.get_position().y1,
                bottom=ax3.get_position().y0,
                hspace=0.3,
                wspace=0.2
            )

            for i, (class_idx, prob) in enumerate(top_classes):
                ax = fig.add_subplot(inner_gs[i])
                
                if class_idx in monkey_cache and monkey_cache[class_idx]:
                    random_img = random.choice(monkey_cache[class_idx])
                    img = Image.open(random_img)
                    ax.imshow(img)
                    ax.set_title(f"{CONFIG['classes'][class_idx]}\n({prob:.2f})", 
                                fontsize=10, pad=5)
                    ax.axis('off')

            for j in range(i+1, n_rows*n_cols):
                fig.add_subplot(inner_gs[j]).axis('off')

        plt.tight_layout()
        original_filename = os.path.basename(row['image_path'])
        save_path = os.path.join(examples_dir, original_filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
        plt.close()
        print(f"Сохранено: {save_path}")

def print_metrics(report, dataset_type):
    """Красивый вывод метрик"""
    print(f"\n{' ' + dataset_type.upper() + ' METRICS ':=^40}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision (Macro): {report['macro avg']['precision']:.4f}")
    print(f"Recall (Macro): {report['macro avg']['recall']:.4f}")
    print(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
   
    print("=" * 40)

def save_model():
    """Сохраняет модель для последующего использования"""
    model.save_pretrained("saved_clip_model")
    processor.save_pretrained("saved_clip_model")
    print("Model saved to saved_clip_model folder")

def main(args):
    """Основной пайплайн выполнения"""
    # Установка delta_look_like в 0, если указан флаг --many_monkeys
    global delta_look_like
    if args.many_monkeys:
        delta_look_like = 0
        print("Режим many_monkeys активирован: delta_look_like установлен в 0")
    
    # Если указан режим people, обрабатываем только людей
    if args.people:
        process_people()
        return
    
    # Если указан режим train, обрабатываем только обучающий набор
    if args.train:
        train_images, _ = load_dataset(CONFIG['data_folders']['train'])
        train_df = classify_images(train_images)
        os.makedirs(os.path.join(CONFIG['results_dir'], 'train'), exist_ok=True)
        train_report = save_metrics(train_df, 'train', CONFIG['results_dir'])
        save_examples(train_df, 'train', CONFIG['results_dir'])
        print_metrics(train_report, 'training')
        return
    
    # Если указан режим validation, обрабатываем только валидационный набор
    if args.validation:
        val_images, _ = load_dataset(CONFIG['data_folders']['validation'])
        val_df = classify_images(val_images)
        os.makedirs(os.path.join(CONFIG['results_dir'], 'validation'), exist_ok=True)
        val_report = save_metrics(val_df, 'validation', CONFIG['results_dir'])
        save_examples(val_df, 'validation', CONFIG['results_dir'])
        print_metrics(val_report, 'validation')
        return
    
    # Если режим не указан, выполняем всё
    train_images, _ = load_dataset(CONFIG['data_folders']['train'])
    train_df = classify_images(train_images)
    os.makedirs(os.path.join(CONFIG['results_dir'], 'train'), exist_ok=True)
    train_report = save_metrics(train_df, 'train', CONFIG['results_dir'])
    save_examples(train_df, 'train', CONFIG['results_dir'])
    
    val_images, _ = load_dataset(CONFIG['data_folders']['validation'])
    val_df = classify_images(val_images)
    os.makedirs(os.path.join(CONFIG['results_dir'], 'validation'), exist_ok=True)
    val_report = save_metrics(val_df, 'validation', CONFIG['results_dir'])
    save_examples(val_df, 'validation', CONFIG['results_dir'])
    
    print_metrics(train_report, 'training')
    print_metrics(val_report, 'validation')
    
    process_people()
    
    save_model()

if __name__ == "__main__":
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Классификация изображений обезьян и людей с помощью CLIP")
    parser.add_argument('--people', action='store_true', help="Обрабатывать только фотографии людей")
    parser.add_argument('--train', action='store_true', help="Обрабатывать только обучающий набор")
    parser.add_argument('--validation', action='store_true', help="Обрабатывать только валидационный набор")
    parser.add_argument('--many_monkeys', action='store_true', help="Установить delta_look_like в 0 для показа большего количества обезьян")
    args = parser.parse_args()
    
    main(args)
    print("\n✅ All processing completed!")