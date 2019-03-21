#В данном файле есть косяк по методике решения(( 
import pandas as pd

#Импортируем данные
df = pd.read_csv("stat2016.csv", header = 0, names = ["absent_days","age","gender"])
#Создаем пустые массивы для мужчин, женщин, взрослых и молодых
male_absent_days = []
female_absent_days = []
elder_absent_days = []
young_absent_days = []
#Заполняем массивы, если число пропущенных дней больше 2
for i in df.index:
    if df.absent_days[i]>2:
        if df.gender[i] == "М":
            male_absent_days.append(df.absent_days[i])
        else:
            female_absent_days.append(df.absent_days[i])
        if df.age[i] > 35:
            elder_absent_days.append(df.absent_days[i])
        else:
            young_absent_days.append(df.absent_days[i])
#Выводим статистическую информацию
print(
    "Статистика по пропускам по болезни за 2016 год.\nВсего пропустившх более двух дней:",
    len(male_absent_days)+len(female_absent_days), "\n",
    "из них мужчин:", len(male_absent_days)," "
    "женщин:", len(female_absent_days), "\n",
    "лица старше 35 лет:", len(elder_absent_days), "\n",
    "лица моложе 35 лет:", len(young_absent_days), "\n",
    "Cредний пропуск(более 2 дней):", "\n",
    "у мужчин:","%.2f" % (sum(male_absent_days)/len(male_absent_days)), "\n",
    "у женщин:","%.2f" % (sum(female_absent_days)/len(female_absent_days)), "\n",
    "у лиц старше 35 лет:", "%.2f" % (sum(elder_absent_days)/len(elder_absent_days)), "\n",
    "у лиц младше 35 лет:", "%.2f" % (sum(young_absent_days)/len(young_absent_days)), "\n"
)
