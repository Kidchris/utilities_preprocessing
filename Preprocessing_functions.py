import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

### dropping unusfull data now
def imputation(data) :
        # if there is problem of data, just decomment these two lines and comment the next one
    #data.drop('Category', axis=1, inplace=True)
    #data.dropna(axis=0, inplace=True)
    data.fillna(0, inplace=True)
    
    return data


## converting string to numbers
def encodage(data):
    for col in data.select_dtypes("object") :
        data[col]= data[col].astype("category").cat.codes
        
    return data


def scaler(data):
    
    min_max = MinMaxScaler()
    data = pd.DataFrame(min_max.fit_transform(data), columns=data.columns)
    
    return data



def preprocess_pipe(data=None):
    
    data = imputation(data)
    data = encodage(data)
    data = scaler(data)
    
    target = data['MSRP']
    features = data.drop('MSRP', axis=1)

    
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42, test_size=0.3)
    return X_train, X_test, y_train, y_test


### MODELISATION


def modelisation(model=None, X_train=None, 
                 y_train=None, X_test=None, 
                 y_test=None):
    
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
    
    
    train_size, train_score, val_score = learning_curve(model, X_train, y_train, train_sizes=np.linspace(0.1, 1., 10))

    plt.figure(figsize=(14,9))
    plt.plot(train_size, train_score.mean(axis=1), label='training')
    plt.plot(train_size, val_score.mean(axis=1), label='validation')
    plt.scatter( [], [], label=model)
    plt.legend()
    plt.show()

def poly_pipeline(model, degre=2, k=13):
    return make_pipeline(PolynomialFeatures
                         (degree=degre, include_bias=False), 
                         SelectKBest(k=k), model)

Linear = poly_pipeline(LinearRegression())


class Preprocess :
    
    def __repr__(self):
        return 'Here we are in class Preprocessing \nType "dir(Preprocess)", to see all alternatives'
    
    ## Preprocess : Splitting in train and test sets
    def split(self, data, target):
        
        self.X = data.drop(target, axis=1)
        self.Y = data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, random_state=0, train_size=0.8)
        print('Train : ',self.X_train.shape,'\nTest : ', self.X_test.shape)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    ## Preprocess : dealing with missing values
    def missings(self, data):
            # if to delete data use dropna from pandas
        data = data.dropna(axis=0)
        return data
            # if need to fill NA values use impute methods
        #from sklearn.impute import SimpleImputer
        #self.model = SimpleImputer(strategy='most_frequent')
        #self.model.fit(data)
        #return self.model.transform(data)
        
    ## Preprocess : converting string to int
    def encoder(self, data):
        for self.dt in data.select_dtypes("object"):
            data[self.dt] = data[self.dt].astype('category').cat.codes
        return data
            # if need for onehotencoding 
        #from sklearn.preprocessing import OneHotEncoder
        #self.model = OneHotEncoder()
        #for dt in data.select_dtypes(include='object'):
            #data[dt] = self.model.fit_transform(data[dt])
    
    ## Preprocess :  Standardisation
    def standar(self, data=None):
        
        self.model = StandardScaler()
        return pd.DataFrame(self.model.fit_transform(data), columns=data.columns)
    
    ## Preprocess : handling outilers and inliers
    def outliers(self,data=None):
            #  in case of needof PCA
        #from sklearn.decomposition import PCA
        #self.model = PCA(n_components=0.95)
        #self.model.fit(data)
        #return self.model.transform(data)
        
        self.model = IsolationForest(contamination= 0.02)
        return self.model.fit_transform(data)
    
    ## Preprocess : in case of need of polynomialFeatures
    def poly_features(self, degre=2):
        
        self.model = PolynomialFeatures(degre)
        return self.model
    
   
    ## Preprocess : regression model just for testing preprocess
    def regression(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None):
        
        self.model = model
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)
        
    
    ## Preprocess : classification model just for testing preprocess
    def classification(self, model=None,  X_train=None, y_train=None, X_test=None, y_test=None):
        
        self.model = model
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)
    
    def curves(self,model=None, X_train=None, y_train=None):
        
        N, train, val = learning_curve(model, X_train, y_train, train_sizes=np.linspace(.1,1.,10))
        plt.plot(N, train.mean(axis=1), label="training")
        plt.plot(N, val.mean(axis=1), label="validation")
        plt.scatter([],[], label=model.__class__.__name__)
        plt.legend()
        
        
# data of float type distributions
def plotFloat(data):
    import seaborn as sb
    from random import choice
    liste =["#900", "yellow", "#0ba", "#330", "#109", '#df8', "#890", "#109", "#FDC", "#011",'#50b']
    length = int(len(data.select_dtypes(exclude='object').columns))
    plt.figure(figsize=(12,10))
    for num,i in zip(data.select_dtypes(exclude='object'), range(1, length+1 )):
        plt.subplot((length//4)+1, 4, i)
        sb.distplot(data[num], color=choice(liste))
    plt.show()
        
def print_objects(df):
    """ Print each object type with its names and features names """
    for types in df.dtypes():
        print(f"Main type of {types}")
        for col in df.select_dtypes(str(types)):
            print(f'Pour la variable {col :-<40} on a les éléments {df[col].unique()}')
            

def plot_objects_pie(df):
    """ Plot all object pielike"""
    p = int(np.sqrt(df.columns[df.dtypes == 'object'].size))+1
    plt.figure(figsize=(20,18))
    i = 0
    for var in df.columns[df.dtypes == 'object']:
        plt.subplot(p, p, i+1)
        df[var].value_counts().plot.pie()
        i += 1
    plt.tight_layout()
    plt.show()