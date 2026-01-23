Olá.
Para a criação desse modelo utilizei uma base de dados que encontrei no Kaggle.
https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images
A base de dados possui 1626 imagens de pulmões de pessoas infectadas com covid-19, 1800 imagens de pulmões com pneumonia e 1802 imagens de pulmões normais.
Para o treinamento e validação separei os dados em 70/30.
Devido à quantidade de imagens, não vou subir as pastas com elas aqui, mas elas estão disponíveis no link acima. 
Possuo as imagens localmente, caso o link saia do ar sinta-se à vontade para entrar em contato e será um prazer te fornece-las.
Ainda estou trabalhando no modelo, ele apresentou overfitting, mesmo apresentandop acurácia de 96%.
Pretendo reduzir a quantidade de épocas (nesse teste usei 10 épocas, o melhor desempenho foi nas épocas 3 e 4).

Atualização 22/01/26:
Não reduzi a quantidade de épocas, reduzi o learning rate e os resultados foram melhores considerando que no primeiro teste a perda na validação ficou entre 0.10 e 0.21, enquanto no segundo a perda oscilou entre 0.09 e 0.11. 

Melhor resultado do primeiro teste (learning rate 0.001):
Época 3/10, 
Perda Treino: 0.0833,
Perda Validação: 0.1092
Tempo por época: 199.51 segundos 

Melhor resultado do segundo teste (learning rate 0.0001:
Época 5/10, 
Perda Treino: 0.0663,
Perda Validação: 0.0969
Tempo por época: 211.23 segundos

A acurácia final foi de 96% em ambos os modelos.

Atualização 23/01/26:
Adicionei um Dropout de 0.5. A acurácia caiu de 96% para 95%, mas aparentemente o overfitting está menor. Vou continuar explorando o modelo e entender quais são as possibilidades de melhoria disponíveis.

Melhor resultado do terceiro teste (learning rate 0.0001 e dropout 0.5)
Época: 9/10,
Perda Treino: 0.0689,
Perda Validação: 0.1086
Tempo por época: 182.61 segundos
Precisão na validação: 95.86%
