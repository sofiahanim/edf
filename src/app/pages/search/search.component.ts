import { Component, OnInit,Output, EventEmitter } from '@angular/core';

@Component({
  selector: 'ngx-search',
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.scss']
})
export class SearchComponent implements OnInit {

  constructor() { }

  ngOnInit(): void {
  }
  
  enteredsearchValue: string = '';
   
  @Output()
  searchValueChange: EventEmitter<string> = new EventEmitter<string>();

  onSearchValueChange(){
    this.searchValueChange.emit(this.enteredsearchValue);
  }

  changeSearchValue(eventData: Event){
    console.log((<HTMLInputElement>eventData.target).value);
    this.enteredsearchValue = (<HTMLInputElement>eventData.target).value;
  }
  
}
