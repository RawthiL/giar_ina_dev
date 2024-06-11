import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import cv2 as cv
from pathlib import Path
import argparse

"""
brief: Given a path of the metadata previously analized, it returns a more balanced dataframe
input: path - string - path to directory of the metadata to analize
output: balanced dataframe
"""
"""
brief: Given a path of the metadata previously analized, it returns the a more balanced dataframe
input: path - string - path to directory of the metadata to analize
output: balanced dataframe
"""
def metadata_balanced(path):
    path = path[:-1] if path.endswith('/') else path # If it ends with / delete it
    #Data loading and initial cleaning

    df=pd.read_csv(path+"/metadata.csv").drop(columns='7')
    mask = (df['0'] == True) & df.loc[:, '1':'6'].eq(False).all(axis=1)
    df=df.drop(index=df[mask].index) #discart images that only contain background

    #Analyze if the dataset is balanced

    dict_len_categories = {}
    for col in df.columns[3:]:  # Exclude the columns 'path' ,'image_name', 0 (background)
        dict_len_categories[col] = df[col].sum() #Total observations per cell type

    total_dict = sum(dict_len_categories.values())
    percentages = np.array(list(dict_len_categories.values())) / total_dict #percentage of each category from 1 to 6
    percentages_dict = {key: percentage for key, percentage in zip(dict_len_categories.keys(), percentages)}

    mean_percentage = np.mean(percentages)
    std_percentage = np.std(percentages)
    cv_percentage = std_percentage / mean_percentage # coefficient of variation CV ,statistical measure used to assess the relative variability of a dataset compared to its mean

    cv_max = 0.4 #Maximum acceptable coefficient of variation (CV)
    per_max = 0.2#Maximum acceptable percentage per category

    check_imbalance = cv_percentage > cv_max or np.any(percentages > per_max) #True for unbalanced, false for balanced.
    print(f"total by category : 1={dict_len_categories['1']}, 2={dict_len_categories['2']} , 3={dict_len_categories['3']}, 4={dict_len_categories['4']}, 5={dict_len_categories['5']}, 6={dict_len_categories['6']}")
    #Balancing dataset

    if check_imbalance == True:
        min_cat = min(percentages_dict, key=percentages_dict.get) #Feature with the least number of observations.
        min_perc = percentages_dict[min_cat] #Percentage of that feature.
        for col in df.columns[3:]:
            perc= df[col].sum()/total_dict  #Calculate the updated percentage after discarding rows for each feature in each cycle of the for loop
            if col !=min_cat and perc > per_max: # Exclude the feature with the lowest percentage and check the maximum percentage
                perc_ajustada = perc-(min_perc+ (perc- min_perc) * 0.5) #Adjustment percentage of the feature decreases as it approaches the minimum value.
                nro_total_drop= int(perc_ajustada*total_dict)  #Total number of rows to remove per category.

                cat=["1","2","3","4","5","6"]
                cat.remove(col)
                mask = (df[col]== True) & (df[min_cat]== False) & (df[col] == True) & (df.loc[:,cat].eq(False).all(axis=1))
                df_tmp=df[mask]

                if len(df_tmp)>=nro_total_drop:
                    indices_to_drop = np.random.choice(df_tmp.index, nro_total_drop, replace=False)
                    df=df.drop(indices_to_drop)
                else:
                    indices_to_drop = np.random.choice(df_tmp.index, len(df_tmp), replace=False)
                    df=df.drop(indices_to_drop)
                    df_tmp2=df[(df[col]== True) & (df[min_cat]== False)]
                    indices_to_drop = np.random.choice(df_tmp2.index, nro_total_drop - len(df_tmp), replace=False)
                    df=df.drop(indices_to_drop)

                dict_len_categories = {} #Update dictionary after the drop operation
                for col in df.columns[3:]:
                    dict_len_categories[col] = df[col].sum()
                total_dict = sum(dict_len_categories.values())
    print(f" updated total by category : 1={dict_len_categories['1']}, 2={dict_len_categories['2']} , 3={dict_len_categories['3']}, 4={dict_len_categories['4']}, 5={dict_len_categories['5']}, 6={dict_len_categories['6']}")
    return df


"""
brief: Given the path and the balanced dataframe, it create the directory and subfolders of train and validation datasets
input: path - string - path to directory of the metadata to analize
input: df - balanced dataframe
output: null
"""

def create_dataset(path,df):

    #Create folder path to store cutted images
    new_path =os.path.dirname(path)+"/balanced_dataset"

    #split the dataset in train and validation
    df['prefix'] = df['image_name'].apply(lambda x: '_'.join(x.split('_')[:2]))
    unique_prefixes = df['prefix'].unique()
    train_prefixes, val_prefixes = train_test_split(unique_prefixes, test_size=0.2, random_state=42)
    train_df = df[df['prefix'].isin(train_prefixes)]
    val_df = df[df['prefix'].isin(val_prefixes)]
    #create directories
    input_train_dir = Path(new_path+"/input/train")
    input_train_dir.mkdir(parents=True, exist_ok=True)
    input_val_dir = Path(new_path+"/input/validation")
    input_val_dir.mkdir(parents=True, exist_ok=True)
    target_train_dir = Path(new_path+"/target/train")
    target_train_dir.mkdir(parents=True, exist_ok=True)
    target_val_dir = Path(new_path+"/target/validation")
    target_val_dir.mkdir(parents=True, exist_ok=True)

    #Updated balanced metadata
    # Crear un DataFrame vacío para almacenar el metadata balanceado
    metadata_balanced = pd.DataFrame(columns=['path', 'image_name', '0', '1', '2', '3', '4', '5', '6'])

    # Listas para almacenar las filas temporales
    rows_train = []
    rows_val = []

    # Iterar sobre las filas de train_df
    for i, v in train_df.iterrows():
        # Leer y guardar la imagen de entrada
        image_input_train = cv.imread(os.path.join(v.iloc[0], "input", v.iloc[1]))
        path_input_train = os.path.join(input_train_dir, v.iloc[1])
        cv.imwrite(path_input_train, image_input_train)

        # Leer y guardar la imagen de destino
        image_target_train = cv.imread(os.path.join(v.iloc[0], "target", v.iloc[1]))
        path_target_train = os.path.join(target_train_dir, v.iloc[1])
        cv.imwrite(path_target_train, image_target_train)

        # Crear un diccionario con los valores a agregar
        new_row = {
            'path': input_train_dir,
            'image_name': v.iloc[1],
            '0': v.iloc[2],
            '1': v.iloc[3],
            '2': v.iloc[4],
            '3': v.iloc[5],
            '4': v.iloc[6],
            '5': v.iloc[7],
            '6': v.iloc[8]
        }

        # Agregar el diccionario a la lista de filas de entrenamiento
        rows_train.append(new_row)

    # Iterar sobre las filas de val_df
    for i, v in val_df.iterrows():
        # Leer y guardar la imagen de entrada
        image_input_val = cv.imread(os.path.join(v.iloc[0], "input", v.iloc[1]))
        path_input_val = os.path.join(input_val_dir, v.iloc[1])
        cv.imwrite(path_input_val, image_input_val)

        # Leer y guardar la imagen de destino
        image_target_val = cv.imread(os.path.join(v.iloc[0], "target", v.iloc[1]))
        path_target_val = os.path.join(target_val_dir, v.iloc[1])
        cv.imwrite(path_target_val, image_target_val)

        # Crear un diccionario con los valores a agregar
        new_row = {
            'path': input_val_dir,
            'image_name': v.iloc[1],
            '0': v.iloc[2],
            '1': v.iloc[3],
            '2': v.iloc[4],
            '3': v.iloc[5],
            '4': v.iloc[6],
            '5': v.iloc[7],
            '6': v.iloc[8]
        }

        # Agregar el diccionario a la lista de filas de validación
        rows_val.append(new_row)

    # Convertir las listas de filas a DataFrames y concatenarlas con metadata_balanced
    metadata_balanced = pd.concat([metadata_balanced, pd.DataFrame(rows_train), pd.DataFrame(rows_val)], ignore_index=True)
    metadata_balanced.to_csv(f"{new_path}/metadata_balanced.csv", index=False)



def main(args):
  if os.path.exists(args.path):
    df = metadata_balanced(args.path)
    if len(df) > 1:
      create_dataset(args.path,df)
    else:
      print("ERROR: No metadata file found on given directory")
  else:
    print("ERROR: Given path is not valid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for creating a balanced dataset")
    parser.add_argument("path", help="Parent directory where the metadata file is located")
    args = parser.parse_args()
    main(args)
