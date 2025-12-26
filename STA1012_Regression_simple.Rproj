---
title: 'STA102 : Regression_simple'
output: pdf_document
date: "2025-12-24"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

Chargement des données :

```{r }
appartements <- read.delim("D:/Etudes/2025_2026/STA102/Regression_simple/appartements.txt")
```

Création du modèle de régression à une variable explicative : 

$$Y=X\beta+\epsilon$$

```{r }
modele=lm(Prix~Superficie, data = appartements)
```

Affichage des paramètres du modèle : 

```{r }
summary(modele)
anova(modele)
predict(modele,interval="confidence",level=0.95)
```

Commentaires :

Significativité du coefficient de régression linéaire : 
P-Value = 7.86e-13


Intervalles de confiance et de prédiction :

```{r }
confint(modele)

modele_appartements = as.data.frame(cbind(
  appartements,
  predict(modele, interval = "confidence", level = 0.95)
))
```

Représentation graphique du modèle :

```{r }
ggplot(appartements, aes(x=Superficie, y=Prix))+ 
  geom_point()+
  geom_smooth(method=lm, se=T)
```
```{r }
PI = as.data.frame(cbind(Prix = appartements$Prix, Superficie = appartements$Superficie,
predict(modele,interval="prediction")))

ggplot(PI, aes(x=Superficie, y=Prix))+
geom_line(aes(y=lwr), color = "red", linetype = "dashed")+
geom_line(aes(y=upr), color = "red", linetype = "dashed")+
geom_point()+geom_smooth(method=lm, se=T) + geom_text(label = row.names(appartements),
vjust = - 1, check_overlap = TRUE, size = 3)
```

Etude des résidus :

```{r }
library(ggfortify)

autoplot(
  modele,
  which = 1,
  ncol = 1,
  label.size = 3,
  label.hjust = -0.8,
  label.n = 6
)

autoplot(
  modele,
  which = 2,
  ncol = 1,
  label.size = 3,
  label.hjust = -0.8,
  label.n = 6
)
```

```{r }
install.packages("olsrr")

library(DescTools)
library(lmtest)
library(olsrr)

# Recherhe d'autocorrélation :
DurbinWatsonTest(modele, alternative = 'two.sided')

# Evaluation de l'homogeneite : 
bptest(modele)

# Test de la normalite :
shapiro.test(modele$residuals)

```
Recherche de valeurs atypiques :

```{r }
autoplot(
  modele,
  which = 3,
  ncol = 1,
  label.size = 3,
  label.hjust = -0.8,
  label.n = 6
) +
  aes(.fitted, .stdresid) + labs(x = "fitted values", y = "standardized residuals") +
  ggtitle("")

# Graphique basique des leviers :
plot(hatvalues(modele))

# Version amélioree du graphique des leviers :
ggplot(modele, aes(seq_along(.hat), .hat)) + geom_col(width = 0.1, colour = "blue") +
  labs(x = "Observation", y = "Leverage") + geom_text(
    label = rownames(appartements),
    check_overlap = T,
    vjust = -0.8,
    size = 3
  ) + geom_hline(yintercept = 4 / nrow(appartements),
                 colour = "red")

# Graphique pour la distznce de cook :  
ols_plot_cooksd_chart(modele)
 

ols_plot_dfbetas(modele)
ols_plot_dffits(modele)
ols_plot_resid_lev(modele)
```
