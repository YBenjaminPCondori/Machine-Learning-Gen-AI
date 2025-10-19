/* The Insertion SOrt ALgorithm in Python */

/* If the element is the first one, it is already sorted.
Move to the next element of the list.
Compare the current element with all elements in the sorted list.
If the element in the sorted list is smaller than the current element, iterate to the next element. Otherwise, shift all the greater element in the list by one position towards the right.
Insert the value at the correct position.
Repeat until the complete list is sorted. */

list=[10,1,5,0,6,8,7,3,11,4]

i=1
while(i<10):
  element=list[i]
  j=i
  i=i+1

  while(j>0 and list[j-1]>indexelement
