import { Component, EventEmitter, OnInit, Output, ElementRef, ViewChild } from '@angular/core';
import { DemoComponent } from '../demo/demo.component';

@Component({
  selector: 'ngx-homepage',
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.scss']

  //add the following line to make the child style similar to the parent
  //encapsulation: ViewEncapsulation.None

  //add the following line in the CHILD component to make the child style NOT similar to the parent
  //encapsulation: ViewEncapsulation.ShadowDom

})
export class HomepageComponent implements OnInit {

  occupation: string = 'designer';

  DisplayNotice(){
    this.display=true;
  }

  title = 'DirectiveExample';
  active: boolean = false;
  videos = [
    {title: 'My video 1', share: 415, likes: 257, dislikes: 12, thumbnail: './../../assets/images/alan.jpg'}, 
    {title: 'My video 2', share: 215, likes: 325, dislikes: 12, thumbnail: './../../assets/images/eva.jpg'}, 
    {title: 'My video 3', share: 513, likes: 105, dislikes: 12, thumbnail: './../../assets/images/jack.jpg'}
  ]

  mostLikedVideo = this.getmostlikedVideo();

  getmostlikedVideo(){
    let videoCopy = [...this.videos];
    return videoCopy.sort((curr, next) => next.likes=curr.likes)[0];
  }


  inputText:string = '';
  destroy: boolean = true;

  DestroyComponent(){  
    this.destroy = false;
  }

  OnSubmit(inputEl: HTMLInputElement){
    this.inputText = inputEl.value;
  }

  constructor() { }

  ngOnInit(): void {
  }

  //title= 'ViewChild';

  @ViewChild('dobInput') dateOfBirth: ElementRef;
  @ViewChild('ageInput') age: ElementRef;
  @ViewChild(DemoComponent,{static:true}) demoComp: DemoComponent;
  
  calculateAge(){
    let birthYear = new Date(this.dateOfBirth.nativeElement.value).getFullYear(); 
    let currentYear = new Date().getFullYear();
    let age = currentYear- birthYear; 
    this.age.nativeElement.value = age;
    // console.log(this.dateOfBirth); 
    // console.log(this.age);
  }

sayHello(inputElement:HTMLInputElement){
  alert('Hello '+ inputElement.value)
  console.log(inputElement)
}

  //radio butto

  //string interpolation: data binding
  slogan: string = 'This is hompage slogan';

  getSlogan(){
    return 'This is function getSlogan';
  }


  //property binding
  source: string = 'assets/images/alan.png';

  display: boolean = false;

  product = [
    {id:1, name:'Eva', image: '../../assets/images/eva.png', type: 'Free'},
    {id:2, name:'Jack', image: '../../assets/images/jack.png',  type: 'Premium'},
    {id:3, name:'Kate', image: '../../assets/images/kate.png', type: 'Free'},
    {id:4, name:'Lee', image: '../../assets/images/lee.png', type: 'Free'},
    {id:5, name:'Nick', image: '../../assets/images/nick.png', type: 'Premium'},
    {id:6, name:'Team', image: '../../assets/images/team.png', type: 'Free'},
    {id:7, name:'Cover', image: '../../assets/images/cover1.jpg', type: 'Premium'}
  
  ];
  
  getTotalCourses(){
    return this.product.length;
  }

  getTotalFreeCourses(){
    return this.product.filter(product => product.type === 'Free').length;
  }

  getTotalPremiumCourses(){
    return this.product.filter(product => product.type === 'Premium').length;
  }

  productCount: string = 'All';

  onFilterChange(data: string){
    this.productCount = data;
    console.log(this.productCount)
  }

  searchValue: string = '';
  onSearchValueEntered(searchValue: string){
    this.searchValue = searchValue;
    console.log(this.searchValue);
  }
}
