import { Component, EventEmitter, OnInit, Output } from '@angular/core';

@Component({
  selector: 'ngx-homepage',
  templateUrl: './homepage.component.html',
  styleUrls: ['./homepage.component.scss']
})
export class HomepageComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
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
