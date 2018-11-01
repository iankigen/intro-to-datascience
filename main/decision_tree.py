from sklearn import tree

print('-' * 10)
print('DECISION TREE')
print('-' * 10)

# [height, weight, shoe size]

x = [
    [181, 80, 44],
    [177, 70, 43],
    [160, 60, 38],
    [154, 54, 37],
    [166, 65, 40],
    [190, 90, 47],
    [175, 64, 39],
    [177, 70, 40],
    [159, 55, 37],
    [171, 75, 42],
    [181, 85, 43],
]

y = ['male', 'female', 'female', 'female',
     'male', 'male', 'male', 'female', 'male',
     'female', 'male']

clf = tree.DecisionTreeClassifier()

clf = clf.fit(x, y)

prediction = clf.predict([[200, 90, 44]])
prediction2 = clf.predict([[150, 75, 38]])

print(prediction2)
print(prediction)
