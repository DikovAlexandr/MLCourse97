## Руководство по Pull Request  

1. Зайдите на страницу [репозитория](http://github.com/DikovAlexandr/MLCourse97/tree/projects) и нажмите `Fork`.  

2. Клонируйте себе репозиторий:  
   ```
   git clone http://github.com/DikovAlexandr/MLCourse97.git
   cd MLCourse97
   ```

3. Добавить оригинальный источник:  
   ```
   git remote add upstream http://github.com/DikovAlexandr/MLCourse97.git
   ```

4. Создать ветку от актуальной `projects`:  
   ```
   git fetch upstream
   git checkout -b my_surname upstream/projects
   ```

5. Внести изменения и закоммитить:  
   ```
   git add .
   git commit -m "Add <Surname> files"
   ```

6. Запушить в свой `Fork`:  
   ```
   git push origin my_surname
   ```

7. Открыть и предложить `Pull Request` в `GitHub`.