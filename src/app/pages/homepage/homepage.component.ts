import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'ngx-homepage',
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.scss']
})
export class HomepageComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }

  //radio button
  all: number = 3;
  free: number = 2;
  premium: number = 1;

  //string interpolation: data binding
  slogan: string = 'This is hompage slogan';

  getSlogan(){
    return 'This is function getSlogan';
  }

  searchValue: string = '';

  changeSearchValue(eventData: Event){
    console.log((<HTMLInputElement>eventData.target).value);
    this.searchValue = (<HTMLInputElement>eventData.target).value;
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
  

}
