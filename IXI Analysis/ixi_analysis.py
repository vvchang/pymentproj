import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import chi2_contingency

def mae(label_path,predict_path):
    """
    Inputs:
        label_path: path to the labels
        predict_path: path to predictions.csv
    Output: 
        Csv of real and predicted age, and BMI, fluid intelligence, and neuroticism (from predictions.csv)
        Prints MAE 
    """
    labels = pd.read_excel(label_path)
    predictions = pd.read_csv(predict_path)

    predictions['IXI_ID'] = predictions['source'].apply(
        lambda path: int(path.split('/')[-1][3:6])
    )
    predictions['age_prediction'] = predictions['age']
    predictions = pd.merge(
        predictions[['IXI_ID', 'age_prediction','sex','bmi','fluid_intelligence','neuroticism']],
        labels[['IXI_ID', 'AGE']],
        on='IXI_ID',
        how='left'
    )

    predictions.dropna(inplace=True)
    predictions['BAG'] = predictions['age_prediction'] - predictions['AGE']
    predictions.to_csv('ixi_predictions2.csv',index=False)

    mae = np.mean(np.abs(predictions['AGE'] - predictions['age_prediction']))
    print(f'MAE: {mae:.2f}')

def age_scatter(csv_file):
    """
    Inputs:
        csv_file: ixi_predictions.csv
    Output:
        Creates scatterplot and linear regression of real vs. predicted age, saved as png
    """
    predictions = pd.read_csv(csv_file)

    m, b = np.polyfit(predictions['AGE'], predictions['age_prediction'], 1)

    sns.regplot(
        x=predictions['AGE'],
        y=predictions['age_prediction'],
        line_kws={"color": "black", "linewidth": 2, "label": f"Regression line: y = {m:.2f}x + {b:.2f}"}
    )

    plt.axline((20, 20), (80, 80), linewidth=2, linestyle='dotted',color='r',label='Unity line')

    plt.xlabel('True age (years)')
    plt.ylabel('Predicted age (years)')
    plt.title('Age prediction on IXI dataset')
    plt.legend()

    plt.savefig("figures/ixi_agepredict.png")

def bag_normal(csv_file):
    """
    Inputs:
        csv_file: ixi_predictions.csv
    Output:
        Creates histogram and Q-Q plot of BAG to visualize if data is normal
    """
    predictions = pd.read_csv(csv_file)

    # Histogram
    plt.figure()
    sns.histplot(predictions['BAG'], bins=30, kde=True)
    plt.xlabel('BAG')
    plt.ylabel('Frequency')
    plt.title('Histogram of BAG')
    plt.savefig("figures/ixi_baghist.png")

    # Q-Q plot
    plt.figure()
    stats.probplot(predictions['BAG'], dist="norm", plot=plt)
    plt.title("Q-Q plot of BAG")
    plt.savefig("figures/ixi_bagqq.png")

def predictphen_bag(csv_file):
    """
    Inputs:
        csv_file: ixi_predictions.csv
    Looks at relationship between BAG and predicted participant phenotypes
    Output: 
        Scatterplots of BMI, fluid intelligence, and neuroticism vs. BAG
    """
    predictions = pd.read_csv(csv_file)
    # BAG vs. BMI
    model = smf.ols("BAG ~ bmi + AGE", data=predictions).fit()
    beta = model.params["bmi"]
    pval = model.pvalues["bmi"]
    r2 = model.rsquared
    print(f"BMI beta = {beta}")
    print(f"p = {pval}")
    print(f"R^2 = {r2}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=predictions, x='bmi', y='BAG')
    plt.title('BAG vs. Predicted BMI with Pearson Correlation (r)')
    plt.xlabel('Predicted BMI')
    plt.ylabel('BAG')
    ax = plt.gca() 
    plt.text(0.05, 0.9, f'r = {r2:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/predict_bmi_bag.png")

    # BAG vs. Fluid intelligence
    model = smf.ols("BAG ~ fluid_intelligence + AGE", data=predictions).fit()
    beta = model.params["fluid_intelligence"]
    pval = model.pvalues["fluid_intelligence"]
    r2 = model.rsquared
    print(f"Fluid intelligence beta = {beta}")
    print(f"p = {pval}")
    print(f"R^2 = {r2}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=predictions, x='fluid_intelligence', y='BAG')
    plt.title('BAG vs. Predicted Fluid Intelligence with Pearson Correlation (r)')
    plt.xlabel('Predicted fluid intelligence')
    plt.ylabel('BAG')
    ax = plt.gca() 
    plt.text(0.05, 0.9, f'r = {r2:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/predict_fluid_intel_bag.png")

    # Neuroticism vs. BAG
    model = smf.ols("BAG ~ neuroticism + AGE", data=predictions).fit()
    beta = model.params["neuroticism"]
    pval = model.pvalues["neuroticism"]
    r2 = model.rsquared
    print(f"Neuroticism beta = {beta}")
    print(f"p = {pval}")
    print(f"R^2 = {r2}")

    plt.figure(figsize=(8, 6))

    m, b = np.polyfit(predictions['neuroticism'], predictions['BAG'], 1)
    sns.regplot(
        x=predictions['neuroticism'],
        y=predictions['BAG'],
        line_kws={"color": "black", "linewidth": 2, "label": f"Regression line: y = {m:.2f}x + {b:.2f}"}
    )
    plt.title('BAG vs. Predicted Neuroticism')
    plt.xlabel('Predicted Neuroticism')
    plt.ylabel('BAG')
    plt.legend()
   
    plt.savefig("figures/predict_neuroticism_bag.png")

def predictphen_age(csv_file):
    """
    Inputs:
        csv_file: ixi_predictions.csv
    Looks at relationship between real age and predicted participant phenotypes
    Output: 
        Scatterplots of BMI, fluid intelligence, and neuroticism vs. age
    """
    predictions = pd.read_csv(csv_file)
    # BMI vs. age
    r_value, p_value = stats.pearsonr(predictions['AGE'], predictions['bmi'])
    print('Predicted BMI vs. age:',p_value)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=predictions, x='AGE', y='bmi')
    plt.title('Predicted BMI vs. Real Age with Pearson Correlation (r)')
    plt.xlabel('Age (years)')
    plt.ylabel('Predicted BMI')
    ax = plt.gca() 
    plt.text(0.05, 0.9, f'r = {r_value:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/predict_bmi_age.png")

    # Fluid intelligence vs. age
    r_value, p_value = stats.pearsonr(predictions['AGE'], predictions['fluid_intelligence'])
    print('Predicted fluid intelligence vs. age:',p_value)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=predictions, x='AGE', y='fluid_intelligence')
    plt.title('Predicted Fluid Intelligence vs. Real Age with Pearson Correlation (r)')
    plt.xlabel('Age (years)')
    plt.ylabel('Predicted fluid intelligence')
    ax = plt.gca() 
    plt.text(0.05, 0.9, f'r = {r_value:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/predict_fluid_intel_age.png")

    # Neuroticism vs. age
    r_value, p_value = stats.spearmanr(predictions['AGE'], predictions['neuroticism'])
    print('Predicted neuroticism vs. age:',p_value)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=predictions, x='AGE', y='neuroticism')
    plt.title('Predicted Neuroticism vs. Real Age with Spearman Correlation (r)')
    plt.xlabel('Age (years)')
    plt.ylabel('Predicted neuroticism')
    ax = plt.gca() 
    plt.text(0.05, 0.95, f'r = {r_value:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/predict_neuroticism_age_sp.png")

def phenotype_real(label_path,predict_path):
    """
    Inputs:
        label_path: path to the labels
        predict_path: path to predictions.csv
    Output: 
        Csv of real and predicted age, and height, weight, ethnic ID, marital ID,
            occupation ID, qualification ID (from IXI label file)
    """
    labels = pd.read_excel(label_path)
    predictions = pd.read_csv(predict_path)

    predictions['IXI_ID'] = predictions['source'].apply(
        lambda path: int(path.split('/')[-1][3:6])
    )
    predictions['age_prediction'] = predictions['age']
    phenotypes = pd.merge(
        predictions[['IXI_ID', 'age_prediction']],
        labels[['IXI_ID', 'AGE','SEX_ID (1=m, 2=f)','HEIGHT','WEIGHT','ETHNIC_ID','MARITAL_ID','OCCUPATION_ID','QUALIFICATION_ID']],
        on='IXI_ID',
        how='left'
    )
    phenotypes.dropna(inplace=True)
    zero_cols = ['SEX_ID (1=m, 2=f)','HEIGHT','WEIGHT','ETHNIC_ID','MARITAL_ID','OCCUPATION_ID','QUALIFICATION_ID']
    phenotypes = phenotypes[(phenotypes[zero_cols] != 0).all(axis=1)]
    phenotypes['BAG'] = phenotypes['age_prediction'] - phenotypes['AGE']
    phenotypes.to_csv('ixi_phenotypes.csv',index=False)

def realphen_plots(csv_file):
    """
    Inputs:
        csv_file: ixi_phenotypes.csv
    Looks at relationship between BAG and real participant phenotypes
    Output: 
        Scatterplots of height, weight, BMI (calculated) vs. BAG
        Boxplots of ethnic, marital, occupation, and qualification IDs vs. BAG
    """
    phenotypes = pd.read_csv(csv_file)
    # Height
    model = smf.ols("BAG ~ HEIGHT + AGE", data=phenotypes).fit()
    beta = model.params["HEIGHT"]
    pval = model.pvalues["HEIGHT"]
    r2 = model.rsquared
    print(f"HEIGHT beta = {beta}")
    print(f"p = {pval}")
    print(f"R^2 = {r2}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=phenotypes, x='HEIGHT', y='BAG')
    plt.title('BAG vs. Height with Pearson Correlation (r)')
    plt.ylabel('BAG (years)')
    plt.xlabel('Height (cm)')
    ax = plt.gca() 
    plt.text(0.05, 0.9, f'r^2 = {r2:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/real_height.png")

    # Weight
    model = smf.ols("BAG ~ WEIGHT + AGE", data=phenotypes).fit()
    beta = model.params["WEIGHT"]
    pval = model.pvalues["WEIGHT"]
    r2 = model.rsquared
    print(f"WEIGHT beta = {beta}")
    print(f"p = {pval}")
    print(f"R^2 = {r2}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=phenotypes, x='WEIGHT', y='BAG')
    plt.title('BAG vs. Weight with Pearson Correlation (r)')
    plt.ylabel('BAG (years)')
    plt.xlabel('Weight (kg)')
    ax = plt.gca() 
    plt.text(0.05, 0.9, f'r^2 = {r2:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/real_weight.png")

    # BMI
    phenotypes['BMI'] = phenotypes['WEIGHT'] / (phenotypes['HEIGHT'] * phenotypes['HEIGHT']) * 10000
    model = smf.ols("BAG ~ BMI + AGE", data=phenotypes).fit()
    beta = model.params["BMI"]
    pval = model.pvalues["BMI"]
    r2 = model.rsquared
    print(f"BMI beta = {beta}")
    print(f"p = {pval}")
    print(f"R^2 = {r2}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=phenotypes['BMI'], y=phenotypes['BAG'])
    plt.title('BAG vs. BMI with Pearson Correlation (r)')
    plt.ylabel('BAG (years)')
    plt.xlabel('BMI')
    ax = plt.gca() 
    plt.text(0.05, 0.9, f'r^2 = {r2:.2f}', transform=ax.transAxes, fontsize=12)
    plt.savefig("figures/real_bmi.png")

    # Sex
    phen = phenotypes.dropna(subset=['SEX_ID (1=m, 2=f)','BAG'])
    phen1 = phen.rename(columns={'SEX_ID (1=m, 2=f)': 'SEX'})
    model = smf.ols("BAG ~ C(SEX) + AGE", data=phen1).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('Sex:',anova_table)

    mapping = {
        1: 'Male',
        2: 'Female'
    }

    phenotypes['SEX_LABEL'] = phenotypes['SEX_ID (1=m, 2=f)'].map(mapping)
    df = phenotypes.dropna(subset=['SEX_LABEL', 'BAG'])
    groups = [group['BAG'].values for _, group in df.groupby('SEX_LABEL')]

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x='SEX_LABEL',
        y='BAG',
        data=df,
        order=list(mapping.values())
    )

    plt.title(f'BAG Grouped by Sex')
    plt.xlabel('Sex')
    plt.ylabel('BAG (years)')

    plt.tight_layout()
    plt.savefig("figures/real_sex.png")

    # Ethnicity
    phen = phenotypes.dropna(subset=['ETHNIC_ID','BAG'])
    model = smf.ols("BAG ~ C(ETHNIC_ID) + AGE", data=phen).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('Ethnicity:',anova_table)

    mapping = {
        1: 'White',
        2: 'Mixed',
        4: 'Black',
        3: 'Asian',
        5: 'Chinese',
        6: 'Other'
    }

    phenotypes['ETHNIC_LABEL'] = phenotypes['ETHNIC_ID'].map(mapping)
    df = phenotypes.dropna(subset=['ETHNIC_LABEL', 'BAG'])
    groups = [group['BAG'].values for _, group in df.groupby('ETHNIC_LABEL')]

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x='ETHNIC_LABEL',
        y='BAG',
        data=df,
        order=list(mapping.values())
    )

    plt.title(f'BAG Grouped by Ethnicity')
    plt.xlabel('Ethnicity')
    plt.ylabel('BAG (years)')

    plt.tight_layout()
    plt.savefig("figures/real_ethnic.png")

    # Marital Status
    phen = phenotypes.dropna(subset=['MARITAL_ID','BAG'])
    model = smf.ols("BAG ~ C(MARITAL_ID) + AGE", data=phen).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('Marital status:',anova_table)

    mapping = {
        1: 'Single',
        2: 'Married',
        4: 'Divorced/Separated',
        3: 'Cohabiting',
        5: 'Widowed'
    }

    phenotypes['MARITAL_LABEL'] = phenotypes['MARITAL_ID'].map(mapping)
    df = phenotypes.dropna(subset=['MARITAL_LABEL', 'BAG'])

    groups = [group['BAG'].values for _, group in df.groupby('MARITAL_LABEL')]

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x='MARITAL_LABEL',
        y='BAG',
        data=df,
        order=list(mapping.values())
    )

    plt.title(f'BAG Grouped by Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('BAG (years)')

    plt.tight_layout()
    plt.savefig("figures/real_marital.png")

    # Occupation
    phen = phenotypes.dropna(subset=['OCCUPATION_ID','BAG'])
    model = smf.ols("BAG ~ C(OCCUPATION_ID) + AGE", data=phen).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('Occupation:',anova_table)

    mapping = {
        1: 'Full-time employment',
        2: 'Part-time employment',
        3: 'Study at college/university',
        4: 'Full-time housework',
        5: 'Retired',
        6: 'Unemployed',
        7: 'Work for pay at home',
        8: 'Other'
    }

    phenotypes['OCCUPATION_LABEL'] = phenotypes['OCCUPATION_ID'].map(mapping)
    df = phenotypes.dropna(subset=['OCCUPATION_LABEL', 'BAG'])

    groups = [group['BAG'].values for _, group in df.groupby('OCCUPATION_LABEL')]

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x='OCCUPATION_LABEL',
        y='BAG',
        data=df,
        order=list(mapping.values())
    )

    plt.title(f'BAG Grouped by Occupation')
    plt.xlabel('Occupation')
    plt.ylabel('BAG (years)')
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig("figures/real_occupation.png")

    # Qualification
    phen = phenotypes.dropna(subset=['QUALIFICATION_ID','BAG'])
    model = smf.ols("BAG ~ C(QUALIFICATION_ID) + AGE", data=phen).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('Qualification:',anova_table)

    mapping = {
        1: 'No qualifications',
        2: 'O-levels, GCSEs, or CSEs',
        3: 'A-levels',
        4: 'Further education e.g. City & Guilds / NVQs',
        5: 'Further education e.g. City & Guilds / NVQs'
    }

    phenotypes['QUALIFICATION_LABEL'] = phenotypes['QUALIFICATION_ID'].map(mapping)
    df = phenotypes.dropna(subset=['QUALIFICATION_LABEL', 'BAG'])

    groups = [group['BAG'].values for _, group in df.groupby('QUALIFICATION_LABEL')]

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        x='QUALIFICATION_LABEL',
        y='BAG',
        data=df,
        order=list(mapping.values())
    )

    plt.title(f'BAG Grouped by Qualificatioin')
    plt.xlabel('Qualification')
    plt.ylabel('BAG (years)')
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig("figures/real_qualification.png")

def compare_sexbmi(csv_predict,csv_real):
    """
    Inputs:
        csv_real: ixi_predictions2.csv
        csv_predict: ixi_phenotypes.csv
    Compares predicted sex and BMI vs. real sex and BMI
    Output: 
        Contingency table of real vs. predicted sex
        Scatterplot of predicted vs. real BMI
    """
    predictions = pd.read_csv(csv_predict)
    predictions["pred_sex"] = np.where(predictions["sex"] < 0.5, 2, 1)
    predictions["pred_bmi"] = predictions["bmi"]
    phenotypes = pd.read_csv(csv_real)
    phenotypes = phenotypes.merge(predictions[["IXI_ID", "pred_sex","pred_bmi"]], on="IXI_ID", how="left")

    # Compare real and predicted sex
    pred = phenotypes['pred_sex']
    real = phenotypes['SEX_ID (1=m, 2=f)']
    combined = pd.concat([real, pred], axis=1)
    combined.columns = ["real", "predicted"]
    combined_clean = combined.dropna()
    contingency_table = pd.crosstab(combined_clean["real"],
                                combined_clean["predicted"])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("Contingency Table:")
    print(contingency_table)
    print("\nChi-squared statistic:", chi2)
    print("p-value:", p)
    print("Degrees of freedom:", dof)
    print("\nExpected frequencies:")
    print(expected)

    # Compare real and predicted BMI 
    phenotypes['BMI'] = phenotypes['WEIGHT'] / (phenotypes['HEIGHT'] * phenotypes['HEIGHT']) * 10000
    phenotypes = phenotypes[phenotypes['BMI'].between(10, 100, inclusive="neither")]
    
    pred = phenotypes['pred_bmi']
    real = phenotypes['BMI']

    r_value, p_value = stats.pearsonr(real, pred)
    print('Predicted BMI vs. real BMI p-value:',p_value)
    print('Predicted BMI vs. real BMI r-value:',r_value)
    mae = np.mean(np.abs(pred - real))          # Mean Absolute Error

    print("MAE:", mae)

    m, b = np.polyfit(real, pred, 1)

    sns.regplot(
        x=real,
        y=pred,
        line_kws={"color": "blue", "linewidth": 2, "label": f"Regression line: y = {m:.2f}x + {b:.2f}"}
    )

    plt.axline((20, 20), (50, 50), linewidth=2, linestyle='dotted',color='r',label='Unity line')

    plt.xlabel('True BMI')
    plt.ylabel('Predicted BMI')
    plt.title('BMI prediction on IXI dataset')
    plt.legend()

    plt.savefig("figures/ixi_bmipredict.png")

def compare_agebag(csv_file):
    """
    Inputs:
        csv_file: ixi_phenotypes.csv
    Compares real age and BAG to see if age is covariate of BAG
    Output: 
        Scatterplot of BAG vs. real age
    """
    phen = pd.read_csv(csv_file)
    r_value, p_value = stats.pearsonr(phen['AGE'], phen['BAG'])
    print(r_value,p_value)
    plt.scatter(phen['AGE'],phen['BAG'])
    plt.xlabel('True Age')
    plt.ylabel('BAG')
    plt.title('BAG vs. True Age')
    plt.savefig('figures/ixi_agebag.png')

# Using each of the above functions for different analyses
label_path = 'pyment-public/data/ixi_all/IXI.xls'
predict_path = 'pyment-public/data/ixi_all/outputs/predictions.csv'
mae(label_path,predict_path)
csv_file1 = 'ixi_predictions2.csv'
age_scatter(csv_file1)
bag_normal(csv_file1)
predictphen_bag(csv_file1)
predictphen_age(csv_file1)
phenotype_real(label_path,predict_path1)
csv_file2 = 'ixi_phenotypes.csv'
realphen_plots(csv_file2)
compare_sexbmi(csv_file1,csv_file2)
