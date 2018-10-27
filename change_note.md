1. Don't put your data in your git repo. Upload them to S3 and download them in your code.
2. Follow PEP8
3. Class name should use CamelCase convension
4. You don't need to inherit from object
5. If you need to add a comment, your function is too big. You should use functions to tell story, not comments
6. Less encoding. No need to shorten a short name. "sent", "leng", "sent_ys" etc. Don't encode this. They don't save time, or look better. 
They only confuse yourself. 
7. Add typing could be useful.
8. if there is "for loop", "if statement", usually inside of them should be a function. i.e. inside for loop and if statement,
 there should be max one line. If statement can also be a function, if it helps you understand what the statement is doing. 
9. It is all about readability. Always think, will you be able to understand a code, if if you haven't worked on it for a while
Make changes that will help you understand/remember what you are doing. 
10. Generator means something in Python. so don't give your Class a name that has a specific meaning in a language. 