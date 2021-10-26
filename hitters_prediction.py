from hitters_pipeline_2 import hitters_data_prep
df = pd.read_csv(r"C:\Users\Kemal\Desktop\veri_bilimi\VBO_BOOTCAMP\datasets\hitters.csv")

X, y = hitters_data_prep(df)

random_user = X.sample(1, random_state=45)
new_model = joblib.load(r"C:\Users\Kemal\Desktop\veri_bilimi\VBO_BOOTCAMP\datasets\voting_reg_hitters.pkl")

new_model.predict(random_user)

