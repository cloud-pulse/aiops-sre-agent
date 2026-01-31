### To print directores as a tree
````
find . | sed 's|[^/]*/|│   |g;s|│   \([^│]*\)$|├── \1|'
````